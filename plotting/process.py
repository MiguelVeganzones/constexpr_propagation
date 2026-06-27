import pathlib
import polars as pl
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
import numpy as np

SUMMARY_SUFFIXES = (
    "_mean",
    "_median",
    "_stddev",
    "_cv",
)

def read_build_csv(path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_csv(path)

    PATTERN = r"(.+?)_(.+?)_([A-Za-z0-9]+)"
    results = (
        df
        .with_columns(
            container=pl.col("name").str.extract(PATTERN, 1).cast(pl.String),
            impl=pl.col("name").str.extract(PATTERN, 2).cast(pl.String),
            name=pl.col("name").str.extract(PATTERN, 3).cast(pl.String),
            # size=pl.col("name").str.extract(PATTERN, 4).cast(pl.Int32),
            # order=pl.col("name").str.extract(PATTERN, 5).cast(pl.Int32),
            build_time=pl.col("build_time").cast(pl.Float32),
        )
    )

    info = pl.read_csv("benchmarks/generated/benchmark_info.csv")

    print(results)
    print(info)
    return results.join(
        info.select(["name", "flops", "output_size"]),
        on="name",
        how="left",
    )


def read_benchmark_csv(path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_csv(path)

    PATTERN = r"^BM_tc_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$"
    results = (
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
            name=pl.col("name").str.extract(PATTERN, 1).cast(pl.String),
            container=pl.col("name").str.extract(PATTERN, 2).cast(pl.String),
            impl=pl.col("name").str.extract(PATTERN, 3).cast(pl.String),
        )
    )

    info = pl.read_csv("benchmarks/generated/benchmark_info.csv")

    print(results)
    print(info)
    return results.join(
        info.select(["name", "flops", "output_size"]),
        on="name",
        how="left",
    )


def load_all(results_dir: pathlib.Path) -> pl.DataFrame:
    dfs = [read_benchmark_csv(p) for p in results_dir.rglob("*.csv")]
    return pl.concat(dfs, how="vertical")


def summary(df: pl.DataFrame):
    s = (
        df
        .group_by([
            "container",
            "impl",
            "flops",
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
    runtime_x: str = "flops",
    runtime_y: str = "real_time",
    metric: str = "median",
) -> pl.DataFrame:

    metric_exprs = {
        "median": pl.col(runtime_y).median(),
        "mean": pl.col(runtime_y).mean(),
        "min": pl.col(runtime_y).min(),
        "max": pl.col(runtime_y).max(),
        "std": pl.col(runtime_y).std(),
    }

    filtered = df.filter(
        (pl.col("impl") == impl)
        & (pl.col("container") == container)
    )

    summary = (
        filtered
        .group_by([runtime_x])
        .agg(metric_exprs[metric].alias("time"))
        .sort([runtime_x])
    )

    return summary


def plot_scaling(
    df: pl.DataFrame,
    runtime_x: str = "flops",
    runtime_y: str = "real_time",
    agg: str = "mean",
    logy: bool = True,
    plot_name = 'scale_plot'
):
    if df.is_empty():
        return

    agg_expr = {
        "median": pl.col(runtime_y).median(),
        "mean": pl.col(runtime_y).mean(),
        "min": pl.col(runtime_y).min(),
        "max": pl.col(runtime_y).max(),
    }[agg]

    summary = (
        df
        .group_by(["impl", "container", runtime_x])
        .agg(agg_expr.alias("time"))
        .sort(["impl", "container", runtime_x])
    )

    pdf = summary.to_pandas()

    combos = sorted(pdf[["impl", "container"]].dropna().drop_duplicates().values.tolist())

    colors = plt.cm.tab10.colors
    color_map = {
        (impl, container): colors[i % len(colors)]
        for i, (impl, container) in enumerate(combos)
    }


    plt.figure(figsize=(6,4))
    for (impl, container), sub in pdf.groupby(["impl", "container"]):
        sub = sub.sort_values(runtime_x)

        label = f"{impl} {container}"

        x = sub[runtime_x]
        y = sub["time"]

        plt.scatter(
            x,
            y,
            color=color_map[(impl, container)],
            s=40,
            label=label
        )

        # plt.plot(
        #     sub[runtime_x],
        #     np.maximum.accumulate(sub["time"]),
        #     color=color_map[(impl, container)],
        #     linestyle='-',
        #     linewidth=1.5,
        # )

        def linear(x, a, b):
            return a * x + b

        xlog = np.log10(x)
        ylog = np.log10(y)

        params, covariance = curve_fit(linear, xlog, ylog)
        a, b = params
        y_fit = 10**linear(xlog, a, b)
        plt.plot(
                x,
                y_fit,
                linewidth=2
                # label=rf"fit: $y={a:.2f}x+{b:.2f}$"
                )

        # plt.plot(
        #     sub[runtime_x],
        #     sub["time"],
        #     color=color_map[(impl, container)],
        #     linestyle='-',
        #     marker="o",
        #     linewidth=1.5,
        #     label=label,
        # )

    plt.xscale("log", base=10)
    if logy:
        plt.yscale("log", base=10)

    if runtime_x == "flops":
        plt.xlabel("Floating-point operations")
    if runtime_x == "footprint":
        plt.xlabel("Memory Footprint [bytes]")
    if runtime_y == "real_time":
        plt.ylabel(f"Runtime ({agg}) [s]")
    if runtime_y == "build_time":
        plt.ylabel(f"Build time ({agg}) [s]")
    plt.title(f"Scaling comparison")

    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(f'report/images/{plot_name}.pdf', transparent=None, dpi='figure',
                format='pdf', bbox_inches='tight')
    plt.show()


def process_benchmarks():
    results_dir = pathlib.Path("results/benchmarking")
    df = load_all(results_dir)
    print(df.head())
    s = summary(df)
    print(s)

    for c in df["impl"].unique():
        for t in df["container"].unique():
            print(f"{t}, {c}")
            m = runtime_matrix(df, t, c, metric="min")
            print(m)
        plot_scaling(df.filter(pl.col("impl")==c), agg='mean', plot_name=f"{c}")
    plot_scaling(df, runtime_x="flops", agg='mean', plot_name='all')
    # plot_scaling(df, runtime_x="footprint")


def process_build_times():
    build_times_file = pathlib.Path("results/compile_report.csv")
    df = read_build_csv(build_times_file)
    print(df.head())
    print(df.columns)
    df = (
        df.group_by([
            "container",
            "impl",
            "flops",
            ]).agg(pl.col("build_time").mean())
        )
    plot_scaling(df, runtime_x="flops", runtime_y="build_time",
                 plot_name="build_times", logy=False)


if __name__ == "__main__":
    process_benchmarks()
   process_build_times()
