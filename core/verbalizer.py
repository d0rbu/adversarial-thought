"""Fixed verbalizer evaluation functions with dtype-aware steering hooks.

This module provides dtype-aware reimplementations of nl_probes verbalizer functions
that properly handle dtype matching to avoid RuntimeError when injecting activations
into models with different dtypes (e.g., float16, bfloat16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch as t

if TYPE_CHECKING:
    from peft import PeftModel

# Import types and utilities from nl_probes
from nl_probes.base_experiment import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    VerbalizerResults,
    collect_target_activations,
    collect_target_responses,
    create_verbalizer_inputs,
    encode_messages,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.dataset_utils import (
    BatchData,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
    )
else:
    from transformers import (  # noqa: TC002
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
    )

# Use our fixed steering hooks
from core.steering_hooks import add_hook, get_hf_activation_steering_hook


@t.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: t.nn.Module,
    tokenizer: AutoTokenizer,
    device: t.device,
    dtype: t.dtype,
    steering_coefficient: float,
    generation_kwargs: dict[str, Any],
) -> list[FeatureResult]:
    """Evaluate features in a batch using dtype-aware steering hooks.

    This is a fixed version that uses our dtype-aware steering hooks from core.steering_hooks.
    """
    batch_steering_vectors = eval_batch.steering_vectors
    batch_positions = eval_batch.positions

    # Create and apply the activation steering hook using our fixed version
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    prompt_tokens = eval_batch.input_ids[:, : eval_batch.input_ids.shape[1]]
    decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)  # type: ignore[attr-defined]

    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)  # type: ignore[attr-defined]

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # type: ignore[attr-defined]

    # Process both samples for each feature consecutively
    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]

        output = decoded_output[i]

        feature_result = FeatureResult(
            feature_idx=feature_idx,
            api_response=output,
            prompt=decoded_prompts[i],
        )
        feature_results.append(feature_result)

    return feature_results


def run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer | AutoTokenizer,
    submodule: t.nn.Module,
    device: t.device,
    dtype: t.dtype,
    global_step: int,  # noqa: ARG001
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict[str, Any],
    verbose: bool = False,
) -> list[FeatureResult]:
    """Run evaluation using dtype-aware steering hooks.

    This is a fixed version that uses our dtype-aware eval_features_batch.
    """
    # Cast tokenizer to AutoTokenizer for nl_probes functions
    auto_tokenizer = cast("AutoTokenizer", tokenizer)

    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:  # type: ignore[attr-defined]
            model.load_adapter(  # type: ignore[attr-defined]
                lora_path,
                adapter_name=adapter_name,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
        model.set_adapter(adapter_name)  # type: ignore[attr-defined]
    with t.no_grad():
        all_feature_results: list[FeatureResult] = []
        for i in tqdm(
            range(0, len(eval_data), eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = get_prompt_tokens_only(e_batch[j])

            # Cast model to PeftModel for materialize_missing_steering_vectors
            # The function expects PeftModel but may work with AutoModelForCausalLM
            peft_model = cast("PeftModel", model)
            e_batch = materialize_missing_steering_vectors(
                e_batch, auto_tokenizer, peft_model
            )

            e_batch = construct_batch(e_batch, auto_tokenizer, device)

            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=auto_tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            if verbose:
                for feature_result in feature_results:
                    print(
                        f"\n=== Feature {feature_result.feature_idx} : {feature_result.api_response} ===\n"
                    )
            all_feature_results.extend(feature_results)

    # Add the meta info to the feature results
    assert len(all_feature_results) == len(
        eval_data
    ), "Number of feature results and evaluation data points must match"
    for feature_result, eval_data_point in zip(
        all_feature_results, eval_data, strict=True
    ):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


def run_verbalizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer_prompt_infos: list[VerbalizerInputInfo],
    verbalizer_lora_path: str | None,
    target_lora_path: str | None,
    config: VerbalizerEvalConfig,
    device: t.device,
    dtype: t.dtype,  # Accept dtype as parameter instead of hardcoding bfloat16
) -> list[VerbalizerResults]:
    """Run verbalizer evaluation with dtype-aware steering hooks.

    This is a fixed version that:
    1. Accepts dtype as a parameter instead of hardcoding torch.bfloat16
    2. Uses our dtype-aware steering hooks from core.steering_hooks

    Assumptions: Both the verbalizer and lora path are LoRA adapters that have already been loaded into the model.
    The lora path's are the `adapter_name` values used when loading the adapters. Both can be None to use the original model.

    This function:
    1. Optionally generates target responses
    2. Collects activations from target LoRA
    3. Runs verbalizer with steering from target activations
    4. Returns structured results
    """
    injection_submodule = get_hf_submodule(model, config.injection_layer)

    if config.add_response_to_context_prompt:
        context_prompts = [ci.context_prompt for ci in verbalizer_prompt_infos]
        context_prompts = collect_target_responses(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            target_lora_path=target_lora_path,
            config=config,
            device=device,
        )
        for i in range(len(verbalizer_prompt_infos)):
            verbalizer_prompt_infos[i].context_prompt = context_prompts[i]

    pbar = tqdm(
        total=len(verbalizer_prompt_infos), desc="Verbalizer Eval Progress", position=1
    )
    results: list[VerbalizerResults] = []

    # Process in activation batches
    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        batch = verbalizer_prompt_infos[start : start + config.eval_batch_size]

        # Build messages and keep combo metadata
        message_dicts: list[list[dict[str, str]]] = []
        combo_bases: list[dict[str, Any]] = []

        for verbalizer_prompt_info in batch:
            correct_answer = verbalizer_prompt_info.ground_truth
            message_dicts.append(verbalizer_prompt_info.context_prompt)

            combo_bases.append(
                {
                    "target_lora_path": target_lora_path,
                    "context_prompt": verbalizer_prompt_info.context_prompt,
                    "verbalizer_prompt": verbalizer_prompt_info.verbalizer_prompt,
                    "ground_truth": correct_answer,
                    "combo_index": start + len(combo_bases),
                }
            )

        # Tokenize as a batch (left padding is configured in load_tokenizer)
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=message_dicts,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )

        target_activations = collect_target_activations(
            model=model,
            inputs_BL=inputs_BL,
            config=config,
            target_lora_path=target_lora_path,
        )

        # Compute per-sample unpadded input_ids and left pad lengths
        seq_len = int(inputs_BL["input_ids"].shape[1])
        context_input_ids_list: list[list[int]] = []

        # Build a single eval batch across all combos and act types
        verbalizer_inputs: list[TrainingDataPoint] = []

        for b_idx in range(len(message_dicts)):
            base = combo_bases[b_idx]
            attn = inputs_BL["attention_mask"][b_idx]
            real_len = int(attn.sum().item())
            left_pad = seq_len - real_len
            context_input_ids = inputs_BL["input_ids"][b_idx, left_pad:].tolist()
            context_input_ids_list.append(context_input_ids)

            for act_key, acts_dict in target_activations.items():
                base_meta = {
                    "target_lora_path": base["target_lora_path"],
                    "context_prompt": base["context_prompt"],
                    "verbalizer_prompt": base["verbalizer_prompt"],
                    "ground_truth": base["ground_truth"],
                    "combo_index": base["combo_index"],
                    "act_key": act_key,
                    "num_tokens": len(context_input_ids),
                    "context_index_within_batch": b_idx,
                }
                verbalizer_inputs.extend(
                    create_verbalizer_inputs(
                        acts_BLD_by_layer_dict=acts_dict,
                        context_input_ids=context_input_ids,
                        verbalizer_prompt=base["verbalizer_prompt"],
                        act_layer=config.active_layer,
                        prompt_layer=config.active_layer,
                        tokenizer=tokenizer,
                        config=config,
                        batch_idx=b_idx,
                        left_pad=left_pad,
                        base_meta=base_meta,
                    )
                )

        if verbalizer_lora_path is not None:
            model.set_adapter(verbalizer_lora_path)  # type: ignore[attr-defined]

        # Run evaluation once for the giant batch using our fixed version
        responses = run_evaluation(
            eval_data=verbalizer_inputs,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,  # Use the provided dtype instead of hardcoded bfloat16
            global_step=-1,
            lora_path=verbalizer_lora_path,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            generation_kwargs=config.verbalizer_generation_kwargs,
        )

        # Aggregate responses per combo and act_key
        agg: dict[tuple[str, int], dict[str, Any]] = {}
        for r in responses:
            meta = r.meta_info
            key = (meta["act_key"], int(meta["combo_index"]))
            if key not in agg:
                agg[key] = {
                    "target_lora_path": target_lora_path,
                    "context_prompt": meta["context_prompt"],
                    "verbalizer_prompt": meta["verbalizer_prompt"],
                    "ground_truth": meta["ground_truth"],
                    "num_tokens": int(meta["num_tokens"]),
                    "context_index_within_batch": int(
                        meta["context_index_within_batch"]
                    ),
                    "token_responses": [None] * int(meta["num_tokens"]),
                    "segment_responses": [],
                    "full_seq_responses": [],
                }
            bucket = agg[key]
            dp_kind = meta["dp_kind"]
            if dp_kind == "tokens":
                idx = int(meta["token_index"])
                bucket["token_responses"][idx] = r.api_response
            elif dp_kind == "segment":
                bucket["segment_responses"].append(r.api_response)
            elif dp_kind == "full_seq":
                bucket["full_seq_responses"].append(r.api_response)
            else:
                raise ValueError(f"Unknown dp_kind: {dp_kind}")

        # Finalize records
        for (act_key, _combo_idx), bucket in agg.items():
            correct_answer = bucket["ground_truth"]
            token_responses = bucket["token_responses"]
            full_sequence_responses = bucket["full_seq_responses"]
            record = VerbalizerResults(
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                context_prompt=bucket["context_prompt"],
                act_key=act_key,
                verbalizer_prompt=bucket["verbalizer_prompt"],
                ground_truth=bucket["ground_truth"],
                num_tokens=bucket["num_tokens"],
                token_responses=token_responses,
                full_sequence_responses=full_sequence_responses,
                segment_responses=bucket["segment_responses"],
                context_input_ids=context_input_ids_list[
                    bucket["context_index_within_batch"]
                ],
            )
            results.append(record)

        if verbalizer_lora_path is not None:
            verbalizer_lora_str = verbalizer_lora_path.split("/")[-1][:40]
        else:
            verbalizer_lora_str = "None"

        if target_lora_path is not None:
            target_lora_str = target_lora_path.split("/")[-1][:40]
        else:
            target_lora_str = "None"

        pbar.set_postfix({"inv": verbalizer_lora_str, "target": target_lora_str})
        pbar.update(len(batch))
    pbar.close()

    return results
