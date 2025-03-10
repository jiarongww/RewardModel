# import sys
# sys.path.append("/u/hpwang/code/RewardModel/ImageReward")

from cProfile import label
import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import requests
from clint.textui import progress
import huggingface_hub
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers.models.align.convert_align_tf_to_hf import preprocess

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer, tokenize
from hpsv2.src.training.train import calc_ImageReward, inversion_score
from hpsv2.src.training.data import ImageRewardDataset, collate_rank, RankingDataset
from hpsv2.utils import root_path, hps_version_map
from ImageReward.ImageReward import ImageReward, BLIPScore, CLIPScore, AestheticScore
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torch


class BenchmarkDataset(Dataset):
    def __init__(self, meta_file, image_folder,transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.open_image = Image.open
        with open(meta_file, 'r') as f:
            prompts = json.load(f)
        used_prompts = []
        files = []
        for idx, prompt in enumerate(prompts):
            filename = os.path.join(self.image_folder, f'{idx:05d}.jpg')
            if os.path.exists(filename):
                used_prompts.append(prompt)
                files.append(filename)
            else:
                print(f"missing image for prompt: {prompt}")
        self.prompts = used_prompts
        self.files = files
            
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        images = self.transforms(self.open_image(img_path))
        caption = self.tokenizer(self.prompts[idx])
        return images, caption

def collate_eval(batch):
    images = torch.stack([sample[0] for sample in batch])
    captions = torch.cat([sample[1] for sample in batch])
    return images, captions

def evaluate_IR(data_path, image_folder, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/ImageReward_test.json'
    dataset = ImageRewardDataset(meta_file, image_folder, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_rank)
    
    score = 0
    total = len(dataset)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, num_images, labels, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            num_images = num_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T * 100
                paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]

            predicted = [torch.argsort(-k) for k in paired_logits_list]
            hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
            labels = [label for label in labels.split(num_images.tolist())]
            score +=sum([calc_ImageReward(paired_logits_list[i].tolist(), labels[i]) for i in range(len(hps_ranking))])
    print('ImageReward:', score/total)

def evaluate_rank(data_path, image_folder, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/test.json'
    dataset = RankingDataset(meta_file, image_folder, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_rank)
    
    score = 0
    total = len(dataset)
    all_rankings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, num_images, labels, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            num_images = num_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T
                paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]

            predicted = [torch.argsort(-k) for k in paired_logits_list]
            hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
            labels = [label for label in labels.split(num_images.tolist())]
            all_rankings.extend(hps_ranking)
            score += sum([inversion_score(hps_ranking[i], labels[i]) for i in range(len(hps_ranking))])
    print('ranking_acc:', score/total)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open('logs/hps_rank.json', 'w') as f:
        json.dump(all_rankings, f)


def evaluate_benchmark(data_path, img_path, model, batch_size, preprocess_val, tokenizer, device):
    meta_dir = data_path
    style_list = os.listdir(img_path)
    model_id = img_path.split('/')[-1]

    score = {}
    
    score[model_id]={}
    for style in style_list:
        # score[model_id][style] = [0] * 10
        score[model_id][style] = []
        image_folder = os.path.join(img_path, style)
        meta_file = os.path.join(meta_dir, f'{style}.json')
        dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images, texts)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
                score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
    print('-----------benchmark score ---------------- ')
    for model_id, data in score.items():
        all_score = []
        for style , res in data.items():
            avg_score = [np.mean(res[i:i+80]) for i in range(0, len(res), 80)]
            all_score.extend(res)
            print(model_id, '{:<15}'.format(style), '{:.2f}'.format(np.mean(avg_score)), '\t', '{:.4f}'.format(np.std(avg_score)))
        print(model_id, '{:<15}'.format('Average'), '{:.2f}'.format(np.mean(all_score)), '\t')

def evaluate_benchmark_all(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device, score_type):
    meta_dir = data_path
    model_list = os.listdir(root_dir)
    style_list = os.listdir(os.path.join(root_dir, model_list[0]))

    score = {}
    for model_id in model_list:
        print(f"\n**********Current Testing Model {model_id}**********")
        score[model_id]={}
        for style in style_list:
            print(f"Current Testing Style {style}")
            # score[model_id][style] = [0] * 10
            score[model_id][style] = []
            image_folder = os.path.join(root_dir, model_id, style)
            meta_file = os.path.join(meta_dir, f'{style}.json')
            if score_type == "hps_v2" or score_type == "hps_v2.1":
                dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        images, texts = batch
                        images = images.to(device=device, non_blocking=True)
                        texts = texts.to(device=device, non_blocking=True)

                        with torch.cuda.amp.autocast():
                            outputs = model(images, texts)
                            image_features, text_features = outputs["image_features"], outputs["text_features"]
                            logits_per_image = image_features @ text_features.T * 100
                        # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
                        score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
            elif score_type in ["imagereward", "BLIP", "CLIP", "Aesthetic"]:
                with open(meta_file, 'r') as f:
                    prompts = json.load(f)
                used_prompts = []
                files = []
                for idx, prompt in enumerate(prompts):
                    filename = os.path.join(image_folder, f'{idx:05d}.jpg')
                    if os.path.exists(filename):
                        used_prompts.append(prompt)
                        files.append(filename)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                imagereward_score = model.score(prompt, filename)
                        score[model_id][style].append(imagereward_score)
                    else:
                        print(f"missing image for prompt: {prompt}")
            elif score_type == "pickscore":
                with open(meta_file, 'r') as f:
                    prompts = json.load(f)
                used_prompts = []
                files = []
                for idx, prompt in enumerate(prompts):
                    filename = os.path.join(image_folder, f'{idx:05d}.jpg')
                    if os.path.exists(filename):
                        used_prompts.append(prompt)
                        files.append(filename)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                # preprocess
                                image_inputs = preprocess_val(
                                    images=[Image.open(filename)],
                                    padding=True,
                                    truncation=True,
                                    max_length=77,
                                    return_tensors="pt",
                                ).to(device)

                                text_inputs = preprocess_val(
                                    text=prompt,
                                    padding=True,
                                    truncation=True,
                                    max_length=77,
                                    return_tensors="pt",
                                ).to(device)
                                # embed
                                image_embs = model.get_image_features(**image_inputs)
                                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                                text_embs = model.get_text_features(**text_inputs)
                                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                                # score
                                pickscore = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                        score[model_id][style].append(float(pickscore))
                    else:
                        print(f"missing image for prompt: {prompt}")



    print('-----------benchmark score ---------------- ')
    results = {}
    for model_id, data in score.items():
        all_score = []
        model_scores = {}
        for style, res in data.items():
            avg_score = [np.mean(res[i:i + 80]) for i in range(0, len(res), 80)]
            all_score.extend(res)
            style_mean = np.mean(avg_score)
            style_std = np.std(avg_score)
            model_scores[style] = {"Mean": style_mean, "Std Dev": style_std}

            print(model_id, '{:<15}'.format(style), '{:.2f}'.format(style_mean), '\t', '{:.4f}'.format(style_std))

        # Calculate overall average and std dev for the model
        overall_mean = np.mean(all_score)
        overall_std = np.std(all_score)
        model_scores["Average"] = {"Mean": overall_mean, "Std Dev": overall_std}

        print(model_id, '{:<15}'.format('Average'), '{:.2f}'.format(overall_mean), '\t')
        results[model_id] = model_scores

    # Save results to JSON file
    output_dir = "./results/" + score_type
    os.makedirs(output_dir, exist_ok=True)

    # Save raw scores to a new JSON file
    raw_scores_file_path = os.path.join(output_dir, "raw_scores.json")
    with open(raw_scores_file_path, "w") as f:
        json.dump(score, f, indent=4)

    json_file_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_file_path}")

    # Prepare data for boxplot
    boxplot_data = {model_id: [] for model_id in score.keys()}
    for model_id, data in score.items():
        for style, res in data.items():
            boxplot_data[model_id].extend(res)

    # Draw boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data.values(), labels=boxplot_data.keys(), patch_artist=True)
    plt.title(f'Model {score_type} Scores Boxplot')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Rotate x-axis labels to make them lean
    plt.xticks(rotation=45)  # Rotate labels by 15 degrees
    # Save the plot to the results directory
    plot_file_path = os.path.join(output_dir, "model_scores_boxplot.png")
    plt.savefig(plot_file_path)
    plt.close()


def evaluate_benchmark_DB(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/drawbench.json'
    model_list = os.listdir(root_dir)

    score = {}
    for model_id in model_list:
        image_folder = os.path.join(root_dir, model_id)
        dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_eval())
        score[model_id] = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images, texts)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                    diag = torch.diagonal(logits_per_image)
                score[model_id] += torch.sum(diag).cpu().item()
            score[model_id] = score[model_id] / len(dataset)
    # with open('logs/benchmark_score_DB.json', 'w') as f:
    #     json.dump(score, f)
    print('-----------drawbench score ---------------- ')
    for model, data in score.items():
        print(model, '\t', '\t', np.mean(data))

model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            None,
            precision=precision,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

        
def evaluate(mode: str, root_dir: str, data_path: str = os.path.join(root_path,'datasets/HPD_v2_test/benchmark'), checkpoint_path: str = None, batch_size: int = 20, score_type: str = "hps_v2") -> None:
    
    # check if the default checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    print('load checkpoint from %s'% checkpoint_path)
    if score_type == "hps_v2" or score_type == "hps_v2.1":
        initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        print(f'Loading model {score_type} ...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer(model_name)
        model = model.to(device)
        model.eval()
        print(f'Loading model {score_type} successfully!')
    elif score_type == "imagereward":
        state_dict = torch.load(checkpoint_path, map_location=device)
        # med_config
        med_config = checkpoint_path.replace("ImageReward.pt", "/med_config.json")
        model = ImageReward(device=device, med_config=med_config).to(device)
        msg = model.load_state_dict(state_dict, strict=False)
        preprocess_val = model.preprocess
        tokenizer = lambda x: model.blip.tokenizer(x, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        model.eval()
        print(f'Loading model {score_type} successfully!')
    elif score_type == "BLIP":
        state_dict = torch.load(checkpoint_path, map_location=device)
        med_config = checkpoint_path.replace("BLIP/model_large.pth", "ImageReward/med_config.json")
        model = BLIPScore(med_config=med_config, device=device).to(device)
        model.blip.load_state_dict(state_dict['model'],strict=False)
        preprocess_val = model.preprocess
        tokenizer = lambda x: model.blip.tokenizer(x, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    elif score_type == "CLIP":
        model = CLIPScore(download_root=checkpoint_path, device=device).to(device)
        preprocess_val = model.preprocess
        tokenizer = lambda x: model.clip_model.tokenizer(x, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    elif score_type == "Aesthetic":
        state_dict = torch.load(checkpoint_path, map_location=device)
        model = AestheticScore(download_root=checkpoint_path.replace("Aesthetic/sac+logos+ava1-l14-linearMSE.pth", "CLIP"), device=device).to(device)
        model.mlp.load_state_dict(state_dict,strict=False)
        preprocess_val = model.preprocess
        tokenizer = lambda x: model.clip_model.tokenizer(x, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    elif score_type == "pickscore":
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        preprocess_val = AutoProcessor.from_pretrained(checkpoint_path.replace("PickScore", processor_name_or_path))
        model = AutoModel.from_pretrained(checkpoint_path).eval().to(device)
        tokenizer = None
    else:
        print(f"Error: score type {score_type} not supported!")
        raise NotImplementedError

    
    if mode == 'ImageReward':
        evaluate_IR(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'test':
        evaluate_rank(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'benchmark_all':
        evaluate_benchmark_all(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device, score_type)
    elif mode == 'benchmark':
        evaluate_benchmark(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'drawbench':
        evaluate_benchmark_DB(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--data-type', type=str, default="benchmark_all", choices=['benchmark', 'benchmark_all', 'test', 'ImageReward', 'drawbench'])
    parser.add_argument('--data-path', type=str, default="./datasets/HPDv2_test/benchmark", help='path to dataset')
    parser.add_argument('--image-path', type=str, default="./datasets/HPDv2_test/benchmark/benchmark_imgs", help='path to image files')
    parser.add_argument('--checkpoint', type=str, default=os.path.join("./pretrained_model/PickScore"), help='path to checkpoint',
                        choices=[os.path.join("./pretrained_model/HPS_v2",'HPS_v2_compressed.pt'),
                                 # https://huggingface.co/xswu/HPSv2
                                 os.path.join("./pretrained_model/HPS_v2.1",'HPS_v2.1_compressed.pt'),
                                 # https://huggingface.co/xswu/HPSv2
                                 os.path.join("./pretrained_model/ImageReward",'ImageReward.pt'),
                                 # https: // huggingface.co / THUDM / ImageReward
                                 os.path.join("./pretrained_model/CLIP"),
                                 # https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
                                 os.path.join("./pretrained_model/Aesthetic", 'sac+logos+ava1-l14-linearMSE.pth'),
                                 # "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
                                 os.path.join("./pretrained_model/BLIP", 'model_large.pth'),
                                 # https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
                                 os.path.join("./pretrained_model/PickScore"),
                                 # https://huggingface.co/yuvalkirstain/PickScore_v1
                                 ])

    parser.add_argument('--score_type', type=str, default="pickscore", help='Score type',
                        choices=["hps_v2", "hps_v2.1", "imagereward", "CLIP", "Aesthetic", "BLIP", "pickscore"])
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()
    
    evaluate(mode=args.data_type, data_path=args.data_path, root_dir=args.image_path, checkpoint_path=args.checkpoint, batch_size=args.batch_size, score_type=args.score_type)

    
    
    
