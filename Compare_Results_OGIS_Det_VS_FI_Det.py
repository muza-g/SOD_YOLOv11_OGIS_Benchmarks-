import os
import sys
import argparse
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_evaluation_metrics(ground_truth_path, predictions_path):
    """
    Evaluate predictions using COCO metrics and extract key metrics.

    Args:
        ground_truth_path (str): Path to the ground truth COCO JSON file.
        predictions_path (str): Path to the predictions COCO JSON file.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    try:
        print(f"Evaluating predictions: {predictions_path} with ground truth: {ground_truth_path}")

        # Load ground truth and predictions
        coco_gt = COCO(ground_truth_path)
        coco_dt = coco_gt.loadRes(predictions_path)

        # Initialize COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

        # Perform evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "AP@[IoU=0.50:0.95]": coco_eval.stats[0],
            "AP@[IoU=0.50]": coco_eval.stats[1],
            "AP@[IoU=0.75]": coco_eval.stats[2],
            "AP@[small]": coco_eval.stats[3],
            "AP@[medium]": coco_eval.stats[4],
            "AP@[large]": coco_eval.stats[5],
            "AR@[IoU=0.50:0.95|max=1]": coco_eval.stats[6],
            "AR@[IoU=0.50:0.95|max=10]": coco_eval.stats[7],
            "AR@[IoU=0.50:0.95|max=100]": coco_eval.stats[8],
            "AR@[small]": coco_eval.stats[9],
            "AR@[medium]": coco_eval.stats[10],
            "AR@[large]": coco_eval.stats[11]
        }

        return metrics

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


def main():
    """
    Main function to calculate results and generate evaluation table.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate evaluation results for Full Inference and GOIS.")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth COCO JSON file.")
    parser.add_argument("--full_inference_path", type=str, required=True, help="Path to the Full Inference JSON file.")
    parser.add_argument("--gois_inference_path", type=str, required=True, help="Path to the GOIS Inference JSON file.")
    args = parser.parse_args()

    # Get metrics for Full Inference
    print("Evaluating Full Inference...")
    full_metrics = get_evaluation_metrics(args.ground_truth_path, args.full_inference_path)

    # Get metrics for GOIS Inference
    print("Evaluating GOIS Inference...")
    gois_metrics = get_evaluation_metrics(args.ground_truth_path, args.gois_inference_path)

    # Generate evaluation results table
    results = []
    for metric, full_value in full_metrics.items():
        gois_value = gois_metrics[metric]

        # Calculate % Improvement
        if full_value != 0:
            improvement = ((gois_value - full_value) / full_value) * 100
            improvement = round(improvement, 2)  # % Improvement rounded to 2 decimal places
        else:
            improvement = "N/A"

        # Append results
        results.append({
            "Metric": metric,
            "Full Inference": round(full_value, 3),  # Round Full Inference to 3 decimal places
            "GOIS Inference": round(gois_value, 3),  # Round GOIS Inference to 3 decimal places
            "% Improvement": improvement
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("\nEvaluation Results:")
    print(results_df)

    # Save results to CSV
    output_path = "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
