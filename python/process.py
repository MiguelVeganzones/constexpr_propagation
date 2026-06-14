import pathlib
import polars as pl
import matplotlib.pyplot as plt
import itertools

SUMMARY_SUFFIXES = (
    "_mean",
    "_median",
    "_stddev",
    "_cv",
)

def read_csv(path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_csv(path)

    PATTERN = r"BM_tc_(\d+)_(\d+)_(\d+)_T(\d+)_C(\d+)"
    return (
        df
        .with_columns(
            source_file=pl.lit(path.name),
        )
        .filter(
            ~pl.col("name").str.ends_with("_mean")
            & ~pl.col("name").str.ends_with("_median")
            & ~pl.col("name").str.ends_with("_stddev")
            & ~pl.col("name").str.ends_with("_cv")
        )
        .with_columns(
            dim=pl.col("name").str.extract(PATTERN, 1).cast(pl.Int32),
            size=pl.col("name").str.extract(PATTERN, 2).cast(pl.Int32),
            order=pl.col("name").str.extract(PATTERN, 3).cast(pl.Int32),
            container=pl.col("name").str.extract(PATTERN, 4).cast(pl.String),
            impl=pl.col("name").str.extract(PATTERN, 5).cast(pl.String),
        )
        .with_columns(
            op_count=(pl.col("size")**(2*pl.col("dim")-pl.col("order"))).cast(pl.Int32),
        )
    )


def load_all(results_dir: pathlib.Path) -> pl.DataFrame:
    dfs = [read_csv(p) for p in results_dir.rglob("*.csv")]
    return pl.concat(dfs, how="vertical")


def summary(df: pl.DataFrame):
    s = (
        df
        .group_by([
            "container",
            "impl",
            "dim",
            "size",
            "order",
            "op_count",
        ])
        .agg(
            pl.col("real_time").median().alias("median_time"),
            pl.col("real_time").mean().alias("mean_time"),
            pl.col("real_time").std().alias("std_time"),
        )
    )
    return s


def runtime_matrix(
    df: pl.DataFrame,
    container: str,
    impl: str,
    runtime_col: str = "real_time",
    metric: str = "median",
) -> pl.DataFrame:

    metric_exprs = {
        "median": pl.col(runtime_col).median(),
        "mean": pl.col(runtime_col).mean(),
        "min": pl.col(runtime_col).min(),
        "max": pl.col(runtime_col).max(),
        "std": pl.col(runtime_col).std(),
    }

    filtered = df.filter(
        (pl.col("impl") == impl)
        & (pl.col("container") == container)
    )

    summary = (
        filtered
        .group_by(["op_count"])
        .agg(metric_exprs[metric].alias("runtime"))
        .sort(["op_count"])
    )

    return summary


def plot_scaling(
    df: pl.DataFrame,
    runtime_col: str = "real_time",
    agg: str = "mean",
    logy: bool = True,
):
    agg_expr = {
        "median": pl.col(runtime_col).median(),
        "mean": pl.col(runtime_col).mean(),
        "min": pl.col(runtime_col).min(),
        "max": pl.col(runtime_col).max(),
    }[agg]

    summary = (
        df
        .group_by(["impl", "container", "dim", "size", "order", "op_count"])
        .agg(agg_expr.alias("runtime"))
        .sort(["impl", "container", "op_count"])
    )

    pdf = summary.to_pandas()

    combos = sorted(pdf[["impl", "container"]].drop_duplicates().values.tolist())

    colors = plt.cm.tab10.colors
    color_map = {
        (impl, container): colors[i % len(colors)]
        for i, (impl, container) in enumerate(combos)
    }

    plt.figure()

    for (impl, container), sub in pdf.groupby(["impl", "container"]):
        sub = sub.sort_values("op_count")

        label = f"T{impl} C{container}"

        plt.plot(
            sub["op_count"],
            sub["runtime"],
            color=color_map[(impl, container)],
            linestyle='-',
            marker="o",
            linewidth=1.5,
            label=label,
        )

    plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=2)
    # plt.ylim(0, 2**33)

    plt.xlabel("Tensor size")
    plt.ylabel(f"Runtime ({agg})")
    plt.title(f"Scaling comparison")

    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_dir = pathlib.Path("results/benchmarking")
    df = load_all(results_dir)
    print(df.head())
    s = summary(df)
    print(s)

    for c in df["impl"].unique():
        for t in df["container"].unique():
            print(f"t{t}, c{c}")
            m = runtime_matrix(df, t, c)
            print(m)
        plot_scaling(df.filter(pl.col("impl")==c))
    plot_scaling(df)
