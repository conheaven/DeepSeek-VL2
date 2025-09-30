import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import glob
import traceback


def get_model_outputs(
    model_path="/beegfs/g3_khw/TDSC/script_tdsc/deepseek/finetuned_models/cc_sbu_align/checkpoint-183",
    device="cuda",
):
    """加载DeepSeek VL2模型和处理器"""
    print(f"Loading model from: {model_path}")

    # 加载处理器
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # 加载模型
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    print("Model and processor loaded.")
    return vl_gpt, vl_chat_processor, tokenizer


def process_inputs_for_generation(model, processor, tokenizer, image_path):
    """处理输入以准备生成"""
    # 构建对话格式
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nDescribe the image.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 加载图像
    pil_images = load_pil_images(conversation)

    # 处理输入
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    return prepare_inputs


def calculate_self_perplexity(scores, generated_ids, min_logit_threshold=None):
    """
    计算模型生成序列的自困惑度 (Self-Perplexity)。
    scores: model.generate() 返回的 logits 元组 (每个元素对应一步生成)
    generated_ids: model.generate() 返回的完整序列 (包含输入 prompt) shape: [batch_size, seq_len]
    min_logit_threshold: 可选的 logits 最小值阈值
    """
    log_probs = []
    num_generated_tokens = len(scores)  # 生成的 token 数量 = scores 的长度

    if num_generated_tokens == 0:
        return 1.0  # 如果没有生成任何 token，perplexity 为 1

    # 确定 prompt 长度
    prompt_len = generated_ids.shape[1] - num_generated_tokens

    for i in range(num_generated_tokens):
        step_logits = scores[i][0].float()  # 获取当前步骤的 logits (batch_size=1)

        # 可选：应用 logit 阈值过滤
        if min_logit_threshold is not None:
            step_logits[step_logits < min_logit_threshold] = min_logit_threshold

        # 计算 log_softmax
        step_log_softmax = F.log_softmax(step_logits, dim=-1)

        # 获取实际生成的 token id
        actual_token_id = generated_ids[0, prompt_len + i].item()

        # 获取该 token 的 log probability
        log_prob = step_log_softmax[actual_token_id].item()

        # 处理过小的对数概率
        MIN_LOG_PROB = -20.0
        log_prob = max(log_prob, MIN_LOG_PROB)
        log_probs.append(log_prob)

    # 计算平均负对数概率
    if not log_probs:
        return 1.0
    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = np.exp(avg_neg_log_prob)

    # 防止 perplexity 变成 inf 或 nan
    if np.isinf(perplexity) or np.isnan(perplexity):
        print(f"Warning: Perplexity became {perplexity}. Clamping to a large value.")
        return 1e10

    return perplexity


def save_generated_distribution_and_perplexity(
    model,
    processor,
    tokenizer,
    inputs,  # 应该是 process_inputs_for_generation 的输出
    save_path="probabilities_mean.txt",
    max_new_tokens=512,  # 控制生成 token 的最大数量
    min_logit_threshold=-10.0,  # Logit 阈值
    top_k_features=120000,  # 最终特征向量维度 (截断用)
):
    """
    使用 model.generate() 生成文本，保存生成过程中平均概率分布和自困惑度。
    """
    # 确保模型在评估模式
    model.eval()

    # 使用 generate 获取生成结果和每一步的 scores (logits)
    with torch.no_grad():
        try:
            # 首先获取输入嵌入
            inputs_embeds = model.prepare_inputs_embeds(**inputs)

            # 使用语言模型生成
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                output_scores=True,  # 获取每一步的 logits
                return_dict_in_generate=True,  # 以字典形式返回结果
                do_sample=False,  # 使用贪婪解码
                use_cache=True
            )
        except Exception as e:
            print(f"Error during model.generate for {save_path}: {e}")
            traceback.print_exc()
            return None

    # 'scores' 是一个元组，包含每个生成步骤的 logits (shape: [batch_size, vocab_size])
    scores = outputs.get("scores", None)
    generated_ids = outputs.get("sequences", None)

    if scores is None or generated_ids is None:
        print(
            f"Warning: generate() did not return scores or sequences for {save_path}."
        )
        return None

    # 计算自困惑度 (Self-Perplexity)
    perplexity = calculate_self_perplexity(scores, generated_ids, min_logit_threshold)

    # 如果没有生成任何 token (scores 为空)
    if not scores:
        print(f"Warning: No tokens generated for {save_path}. Saving default vector.")
        # 使用tokenizer的词汇表大小
        vocab_size = len(tokenizer)
        mean_probs_np = np.zeros(vocab_size)
        num_generated = 0
    else:
        num_generated = len(scores)
        # 计算每个生成步骤的概率分布并求平均
        all_step_probabilities = []
        vocab_size = scores[0].shape[-1]  # 从第一个 score 获取词汇表大小
        for step_logits in scores:
            # 确保 logits 在 CPU 上并且是 float 类型进行 softmax 计算
            step_logits_processed = (
                step_logits[0].float().cpu()
            )  # 获取 batch_size=1 的 logits

            # 应用 logit 阈值过滤
            if min_logit_threshold is not None:
                step_logits_processed[step_logits_processed < min_logit_threshold] = (
                    min_logit_threshold
                )

            # 计算概率分布
            step_probs = torch.softmax(step_logits_processed, dim=0)
            all_step_probabilities.append(step_probs)

        # 堆叠并计算平均概率分布
        if all_step_probabilities:
            mean_probabilities = torch.mean(torch.stack(all_step_probabilities), dim=0)
            mean_probs_np = mean_probabilities.numpy()
        else:
            mean_probs_np = np.zeros(vocab_size)

    # 截断或处理 mean_probs_np 以匹配 top_k_features
    current_vocab_size = len(mean_probs_np)
    if current_vocab_size >= top_k_features:
        # 如果词汇表大小足够，直接取前 top_k_features 个概率值
        feature_probs = mean_probs_np[:top_k_features]
    else:
        # 如果词汇表大小不足 top_k_features，用 0 填充
        print(
            f"Warning: Vocab size {current_vocab_size} < top_k_features {top_k_features} for {save_path}. Padding with zeros."
        )
        feature_probs = np.zeros(top_k_features)
        feature_probs[:current_vocab_size] = mean_probs_np

    # 添加 perplexity 到特征向量末尾
    combined_features = np.append(feature_probs, perplexity)

    # 保存结合了 perplexity 的特征向量
    try:
        np.savetxt(save_path, combined_features, fmt="%.10f")
    except Exception as e:
        print(f"Error saving features to {save_path}: {e}")
        traceback.print_exc()
        return None

    # 返回信息
    return {
        "mean_probabilities_feature": feature_probs,
        "perplexity": perplexity,
        "num_generated": num_generated,
        "save_path": save_path,
    }


def process_image(image_path, model, processor, tokenizer, img_num, output_dir):
    """处理单张图片并保存基于生成的概率分布和perplexity"""
    try:
        # 1. 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False

        # 2. 尝试打开图片验证格式
        try:
            with Image.open(image_path) as test_img:
                test_img.verify()
        except Exception as e:
            print(f"Error validating image {image_path}: {e}")
            return False

        # 3. 处理输入
        try:
            inputs = process_inputs_for_generation(model, processor, tokenizer, image_path)
        except Exception as e:
            print(f"Error processing inputs for {image_path}: {e}")
            traceback.print_exc()
            return False

        # 4. 保存基于生成的概率分布和 perplexity
        output_path = os.path.join(output_dir, f"file_{img_num}.txt")
        result = save_generated_distribution_and_perplexity(
            model,
            processor,
            tokenizer,
            inputs,
            output_path,
            max_new_tokens=512,
            top_k_features=120000,
        )

        # 检查保存函数是否成功返回结果
        if result is None:
            print(
                f"Failed to save features for {image_path}. Skipping perplexity summary."
            )
            return False

        # 5. 记录 perplexity 到汇总文件
        perplexity_summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        image_name = os.path.basename(image_path)

        # 检查汇总文件是否存在，不存在则创建并添加标题行
        if not os.path.exists(perplexity_summary_path):
            try:
                with open(perplexity_summary_path, "w") as f:
                    f.write("image_name,self_perplexity,num_generated_tokens\n")
            except Exception as e:
                print(
                    f"Error creating perplexity summary file {perplexity_summary_path}: {e}"
                )

        # 添加当前图像的 perplexity 记录
        try:
            with open(perplexity_summary_path, "a") as f:
                f.write(
                    f"{image_name},{result['perplexity']:.10f},{result['num_generated']}\n"
                )
        except Exception as e:
            print(
                f"Error writing to perplexity summary file {perplexity_summary_path}: {e}"
            )

        return True

    except Exception as e:
        print(f"Unexpected error processing image {image_path}: {str(e)}")
        traceback.print_exc()
        return False


def process_directory(
    base_input_dir, base_output_dir, split, condition, model, processor, tokenizer
):
    """处理指定目录下的所有图片"""
    # 构建输入和输出目录路径
    input_dir = os.path.join(base_input_dir, split, condition)
    output_dir = os.path.join(base_output_dir, split, condition)

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}. Skipping.")
        return 0

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return 0

    # 获取所有图片路径
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_paths:
        print(
            f"No images found in {input_dir} with patterns {image_patterns}. Skipping."
        )
        return 0

    print(f"\nProcessing {split}/{condition}")
    print(f"Found {len(image_paths)} images in {input_dir}")

    # 清理旧的 perplexity 汇总文件
    perplexity_summary_path = os.path.join(output_dir, "perplexity_summary.csv")
    if os.path.exists(perplexity_summary_path):
        print(f"Removing old summary file: {perplexity_summary_path}")
        os.remove(perplexity_summary_path)

    # 批量处理图片
    successful = 0
    for img_num, image_path in tqdm(
        enumerate(image_paths),
        total=len(image_paths),
        desc=f"Processing {split}/{condition}",
    ):
        if process_image(
            image_path, model, processor, tokenizer, img_num, output_dir
        ):
            successful += 1

    print(
        f"Successfully processed {successful}/{len(image_paths)} images for {split}/{condition}"
    )
    return successful


def main():
    # 配置区
    BASE_INPUT_DIR = "/beegfs/g3_khw/TDSC/datasets/mlp/cc_sbu_align"  # 需要修改为实际输入路径
    BASE_OUTPUT_DIR = "/beegfs/g3_khw/TDSC/script_tdsc/deepseek/main_exp/cc_sbu_align"  # 输出路径
    MODEL_PATH = "/beegfs/g3_khw/TDSC/script_tdsc/deepseek/finetuned_models/cc_sbu_align/checkpoint-183"

    # 定义需要处理的目录组合
    SPLITS = ["train", "test"]
    CONDITIONS = ["finetuned","unfinetuned"]

    print("Initializing DeepSeek VL2 model and processor...")
    try:
        model, processor, tokenizer = get_model_outputs(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize model or processor: {e}")
        traceback.print_exc()
        return

    # 总处理计数
    total_processed = 0
    total_images_found = 0

    # 处理每个目录组合
    for split in SPLITS:
        for condition in CONDITIONS:
            # 获取该目录下的总图片数
            current_input_dir = os.path.join(BASE_INPUT_DIR, split, condition)
            if os.path.isdir(current_input_dir):
                image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                current_images = 0
                for pattern in image_patterns:
                    current_images += len(
                        glob.glob(os.path.join(current_input_dir, pattern))
                    )
                total_images_found += current_images
            else:
                current_images = 0

            # 调用处理函数
            successful_count = process_directory(
                BASE_INPUT_DIR, BASE_OUTPUT_DIR, split, condition, model, processor, tokenizer
            )
            total_processed += successful_count

    print(f"\nProcessing complete!")
    print(
        f"Total images found across all specified input directories: {total_images_found}"
    )
    print(f"Total images successfully processed (features saved): {total_processed}")
    print(f"Features saved to base directory: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()