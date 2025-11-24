import time

import cv2
import numpy as np
import torch

from kestrel.third_party.hqsam.segment_anything.utils.transforms import (
    ResizeLongestSide,
)

from .utils import extract_bboxes_expand, extract_mask, extract_points, prepare_image


def _build_sam_inputs(
    image: torch.Tensor,
    pred_masks: torch.Tensor,
    image_embeddings: torch.Tensor,
    resize_transform: ResizeLongestSide,
    *,
    use_point: bool,
    use_box: bool,
    use_mask: bool,
    add_neg: bool,
    margin: float,
    gamma: float,
    strength: float,
    boxes_xyxy: torch.Tensor | None,
) -> dict:
    """Prepare prompt dict for SAM-HQ."""

    ori_size = pred_masks.shape[-2:]
    input_dict = {"image": image, "original_size": ori_size}

    target_size = image.shape[1:]
    expand_list = torch.zeros((len(pred_masks)), device=image.device)

    if use_box:
        if boxes_xyxy is not None:
            bboxes = boxes_xyxy.to(image.device)
        else:
            bboxes, _, _, expand_list = extract_bboxes_expand(
                image_embeddings, pred_masks, margin=margin
            )
        input_dict["boxes"] = resize_transform.apply_boxes_torch(bboxes, ori_size)

    point_coords, point_labels, gaus_dt = extract_points(
        pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma
    )
    if use_point:
        input_dict["point_coords"] = resize_transform.apply_coords_torch(
            point_coords, ori_size
        )
        input_dict["point_labels"] = point_labels

    if use_mask:
        input_dict["mask_inputs"] = extract_mask(
            pred_masks,
            gaus_dt,
            target_size,
            is01=True,
            strength=strength,
            device=image.device,
            expand_list=expand_list,
        )

    return input_dict


def sam_refiner(
    image_bgr: np.ndarray,
    coarse_masks: np.ndarray | list[np.ndarray],
    sam,
    *,
    resize_transform: ResizeLongestSide | None = None,
    use_point: bool = True,
    use_box: bool = True,
    use_mask: bool = True,
    add_neg: bool = True,
    iters: int = 5,
    margin: float = 0.0,
    gamma: float = 4.0,
    strength: float = 30.0,
    ddp: bool = False,
    is_train: bool = False,
    timing: dict | None = None,
    boxes_xyxy: torch.Tensor | None = None,
):
    """HQ-only SAM refiner."""

    timing = {} if timing is None else timing
    total_start = time.perf_counter()

    masks_np = np.stack(coarse_masks, axis=0) if isinstance(coarse_masks, list) else coarse_masks
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    coarse_masks_t = torch.tensor(masks_np, dtype=torch.uint8, device=sam.device)

    if resize_transform is None:
        target = sam.module.image_encoder.img_size if ddp else sam.image_encoder.img_size
        resize_transform = ResizeLongestSide(target)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = [prepare_image(image_rgb, resize_transform, sam.device)]

    with torch.no_grad():
        enc_start = time.perf_counter()
        if ddp:
            input_images = torch.stack([sam.module.preprocess(x) for x in image_tensor], dim=0)
            image_embeddings, interm_embeddings = sam.module.image_encoder(input_images)
            interm_embeddings = interm_embeddings[0]
        else:
            input_images = torch.stack([sam.preprocess(x) for x in image_tensor], dim=0)
            image_embeddings, interm_embeddings = sam.image_encoder(input_images)
            interm_embeddings = interm_embeddings[0]
        timing["image_encoding_s"] = time.perf_counter() - enc_start

    iter_times: list[float] = []
    sam_masks_list = coarse_masks_t
    for _ in range(iters):
        iter_start = time.perf_counter()
        input_dict = _build_sam_inputs(
            image_tensor[0],
            sam_masks_list,
            image_embeddings,
            resize_transform,
            use_point=use_point,
            use_box=use_box,
            use_mask=use_mask,
            add_neg=add_neg,
            margin=margin,
            gamma=gamma,
            strength=strength,
            boxes_xyxy=boxes_xyxy,
        )

        sam_input = [input_dict]

        if ddp:
            sam_output = sam.module.forward_with_image_embeddings(
                image_embeddings,
                interm_embeddings,
                sam_input,
                multimask_output=True,
            )[0]
        else:
            sam_output = sam.forward_with_image_embeddings(
                image_embeddings,
                interm_embeddings,
                sam_input,
                multimask_output=True,
            )[0]

        sam_masks = sam_output["masks"]
        sam_masks3 = sam_masks.clone().detach()
        sam_ious = sam_output["iou_predictions"]
        if is_train:
            return sam_masks, sam_ious, sam_masks3

        top_masks = []
        for mask_stack, iou_stack in zip(sam_masks, sam_ious):
            max_idx = torch.argmax(iou_stack)
            top_masks.append(mask_stack[max_idx])

        sam_masks = torch.stack(top_masks, dim=0)
        sam_masks_list = sam_masks > 0
        iter_times.append(time.perf_counter() - iter_start)

    refined_masks = sam_masks_list.cpu().numpy().astype(np.uint8)
    timing["iter_s"] = iter_times
    timing["total_refine_s"] = time.perf_counter() - total_start
    return refined_masks, sam_ious, sam_masks3, timing
