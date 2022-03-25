import argparse
import json

import jax
import jax.numpy as jnp
import numpy as onp

from meta_expl.utils import (
    accuracy,
    neg_rmse,
    pearson,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_files", nargs="+", help="model output files")
    parser.add_argument("--metric", default="accuracy", help="metric to use")
    parser.add_argument("--num-bootstraps", type=int, default=500)
    parser.add_argument("--sample-ratio", default=0.5)
    args = parser.parse_args()

    num_systems = len(args.output_files)

    if args.metric == "accuracy":
        metric = accuracy
    elif args.metric == "neg_rmse":
        metric = neg_rmse
    elif args.metric == "pearson":
        metric = pearson
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    metric = jax.jit(metric)

    all_outputs = []
    all_ground_truths = []
    for outfile in args.output_files:
        with open(outfile, "r") as f:
            models_outputs = []
            ground_truths = []
            for line in f:
                obj = json.loads(line)
                models_outputs.append(obj["y_hat"])
                ground_truths.append(obj["y"])

        all_outputs.append(models_outputs)
        all_ground_truths.append(ground_truths)

    # check if all ground truths are the same
    for i in range(len(all_ground_truths[0])):
        for j in range(1, len(all_ground_truths)):
            if metric == "person":
                assert all(
                    onp.isclose(
                        all_ground_truths[0][i], all_ground_truths[j][i], atol=0.5e-2
                    )
                )
            if metric == "accuracy":
                assert all_ground_truths[0][i] == all_ground_truths[j][i]

    ground_truths = all_ground_truths[0]

    metric_values = []
    for sys in range(num_systems):
        metric_values.append(
            metric(jnp.array(all_outputs[sys]), jnp.array(ground_truths))
        )
    mean = onp.mean(metric_values)
    median = onp.median(metric_values)
    min_metric, max_metric = onp.min(metric_values), onp.max(metric_values)
    std = onp.std(metric_values)
    q75, q25 = onp.percentile(metric_values, [75, 25])

    ids = list(range(len(ground_truths)))
    num_points = int(len(ids) * args.sample_ratio)
    all_means = []

    for b in range(args.num_bootstraps):
        reduced_ids = onp.random.choice(ids, size=num_points, replace=True)
        metric_value = 0
        for sys in range(num_systems):
            reduced_outputs = jnp.stack(
                [jnp.array(all_outputs[sys][i]) for i in reduced_ids]
            )
            reduced_ground_truths = jnp.stack(
                [jnp.array(ground_truths[i]) for i in reduced_ids]
            )
            metric_value += metric(reduced_outputs, reduced_ground_truths)
        metric_value /= num_systems
        all_means.append(metric_value)

    bootstrapped_mean = onp.mean(all_means)
    conf_interval = onp.percentile(all_means, [2.5, 97.5])

    print(args.metric)
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Min/Max: {min_metric}/{max_metric}")
    print(f"q25/q75: {q25}/{q75}")
    print(f"STD: {std}")
    print(f"Bootstrapped Mean: {bootstrapped_mean}")
    print(f"Confidence Interval: [{conf_interval[0]}, {conf_interval[1]}]")


if __name__ == "__main__":
    main()
