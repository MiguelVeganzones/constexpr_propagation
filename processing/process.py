import pathlib
import argparse
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


def object_size_df(preset) -> pl.DataFrame:
    tmp_build_dir = pathlib.Path(f"build/{preset}/tmp")
    sizes_df = pl.DataFrame(
        {
            "name": [p.stem for p in tmp_build_dir.glob("*.o")],
            "object_size": [p.stat().st_size for p in tmp_build_dir.glob("*.o")],
        }
    )
    PATTERN = r"(.+?)_(.+?)_([A-Za-z0-9]+)"
    sizes_df = (
        sizes_df
        .with_columns(
            container=pl.col("name").str.extract(PATTERN, 1).str.to_uppercase(),
            impl=pl.col("name").str.extract(PATTERN, 2).str.to_uppercase(),
            name=pl.col("name").str.extract(PATTERN, 3),
        )
    )
    print(sizes_df)
    return sizes_df


def read_build_csv(path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_csv(path)

    PATTERN = r"(.+?)_(.+?)_([A-Za-z0-9]+)"
    results = (
        df
        .with_columns(
            container=pl.col("name").str.extract(PATTERN, 1).cast(pl.String),
            impl=pl.col("name").str.extract(PATTERN, 2).cast(pl.String),
            name=pl.col("name").str.extract(PATTERN, 3).cast(pl.String),
            build_time=pl.col("build_time").cast(pl.Float32),
        )
    )

    info = pl.read_csv("benchmarks/generated/benchmark_info.csv")

    print(results)
    print(info)
    return (
        results
        .join(
            info.select(["name", "flops", "output_size"]),
            on="name",
            how="left",
        )
    )


def read_benchmark_csv(path: pathlib.Path, preset) -> pl.DataFrame:
    df = pl.read_csv(path)
    print(df)
    PATTERN = r"^BM_tc_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$"
    results = (
        df
        .with_columns(
            source_file=pl.lit(path.name),
            real_time=pl.col("real_time").cast(pl.Float64),
        )
        .select(["name","iterations","real_time"])
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
    sizes_df = object_size_df(preset)

    print(results)
    print(info)
    return (
        results
        .join(
            info.select(["name", "flops", "output_size"]),
            on="name",
            how="left",
        )
        .join(
            sizes_df.select(["container", "impl", "name", "object_size"]),
            on=["container", "impl", "name"],
            how="left",
        )
    )


def load_all(preset) -> pl.DataFrame:
    results_dir = pathlib.Path(f"results/{preset}/benchmarking")
    baseline_dir = pathlib.Path(f"results/benchmarking")
    dfs = [read_benchmark_csv(p, preset) for d in [results_dir, baseline_dir] for p in d.rglob("*.csv")]
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
    plot_name = 'scale_plot',
    line = 'log',
    unit = 1e-9
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
        print(f"{impl}, {container}, {sub["time"]}")
        y = sub["time"] * unit

        plt.scatter(
            x,
            y,
            color=color_map[(impl, container)],
            s=40,
            label=label
        )

        if line == 'log':
            def linear(x, a, b):
                return a * x + b

            xlog = np.log10(x)
            ylog = np.log10(y)

            params, covariance = curve_fit(linear, xlog, ylog)
            a, b = params
            y_fit = 10**linear(xlog, a, b)
            print(xlog)
            print(y_fit)
            plt.plot(
                    x,
                    y_fit,
                    linestyle='-',
                    linewidth=2
                    # label=rf"fit: $y={a:.2f}x+{b:.2f}$"
                    )
        elif line == 'max':
            plt.plot(
                x,
                np.maximum.accumulate(y),
                color=color_map[(impl, container)],
                linestyle='-',
                linewidth=2,
            )

    plt.xscale("log", base=10)
    if logy:
        plt.yscale("log", base=10)

    if runtime_x == "flops":
        plt.xlabel("Floating-point operations")
    if runtime_x == "footprint":
        plt.xlabel("Memory Footprint [bytes]")
    if runtime_y == "real_time":
        plt.ylabel(f"Runtime ({agg}) [s]")
        plt.title(f"Runtime scaling")
    if runtime_y == "build_time":
        plt.ylabel(f"Build time ({agg}) [s]")
        plt.title(f"Build time scaling")
    if runtime_y == "object_size":
        plt.ylabel(f"Object size ({agg}) [KB]")
        plt.ylim(0, 30)
        plt.title(f"Object size scaling")

    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(f'report/images/{plot_name}.pdf', transparent=None, dpi='figure',
                format='pdf', bbox_inches='tight')
    plt.show()


def process_benchmarks(preset):
    df = load_all(preset)
    print(df.head())
    s = summary(df)
    print(s)
    aggm = 'median'
    for c in df["impl"].unique():
        for t in df["container"].unique():
            print(f"{t}, {c}")
            m = runtime_matrix(df, t, c, metric="min")
            print(m)
        plot_scaling(df.filter(pl.col("impl")==c), agg=aggm, plot_name=f"{c}")
    plot_scaling(df, runtime_x="flops", agg=aggm, plot_name='all')
    plot_scaling(df.filter(pl.col('container').str.slice(1,1) ==
                           pl.col('impl').str.slice(1,1)), runtime_x="flops",
                 agg=aggm, plot_name='runtime')
    # plot_scaling(df, runtime_x="footprint")


def process_build_times(preset):
    build_times_file = pathlib.Path(f"results/{preset}/compile_report.csv")
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
                 plot_name="build_times", logy=False, line='max', unit = 1)


def process_build_sizes(preset):
    df = load_all(preset)
    print(df.head())
    print(df.columns)
    df = (
        df.group_by([
            "container",
            "impl",
            "flops",
            ]).agg(pl.col("object_size").mean())
        ).filter(pl.col('container') != 'np')
    plot_scaling(df, runtime_x="flops", runtime_y="object_size",
                 plot_name="build_sizes", logy=False, line='max', unit = 1/1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preset")
    preset = parser.parse_args().preset

    process_build_sizes(preset)
    process_benchmarks(preset)
    process_build_times(preset)
