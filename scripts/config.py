from dataclasses import dataclass
from typing import Tuple, List
import random
import math


# =========================================================
# LIMITS
# =========================================================

BENCHMARK_MIN_RUNTIME = 2
BENCHMARK_ITERATIONS = 10
BENCHMARK_WARMUPS = 2
COMPILE_ITERATIONS = 2

MAX_FLOPS = 10**11
MAX_MEMORY = 2**30

SIZES = [
    [2, 4, 8, 16, 32, 64, 128, 256],
    [3, 7, 15, 31, 63, 127, 255],
]

RANKS = [
    [2, 3, 4, 5, 6, 7, 8, 16, 32, 64],
    [3, 4, 5, 6, 13, 19],
]

FLOPS_FILTER_DISTANCE = 0.2
FLOPS_FILTER_REJECTION_PROBABILITY = 0.3

random.seed(25236)

# =========================================================
# CASE MODEL
# =========================================================

@dataclass(frozen=True)
class Case:
    name: str
    a_shape: Tuple[int, ...]
    b_shape: Tuple[int, ...]
    cis: Tuple[Tuple[int, int], ...]
    dtype_bytes: int = 4   # float32

    @property
    def a_size(self):
        return math.prod(self.a_shape)

    @property
    def b_size(self):
        return math.prod(self.b_shape)

    @property
    def output_shape(self):
        contracted_a = {
            i for i, _ in self.cis
        }
        contracted_b = {
            j for _, j in self.cis
        }
        return (
            tuple(
                self.a_shape[i]
                for i in range(len(self.a_shape))
                if i not in contracted_a
            )
            +
            tuple(
                self.b_shape[j]
                for j in range(len(self.b_shape))
                if j not in contracted_b
            )
        )

    @property
    def output_size(self):
        return math.prod(self.output_shape)

    @property
    def reduction_size(self):
        return math.prod(
            self.a_shape[i]
            for i, _ in self.cis
        )

    @property
    def flops(self):
        multiplies = (
            self.output_size *
            self.reduction_size
        )
        # multiply + accumulate
        return 2 * multiplies

    @property
    def memory_bytes(self):
        a_bytes = (
            self.a_size *
            self.dtype_bytes
        )
        b_bytes = (
            self.b_size *
            self.dtype_bytes
        )
        c_bytes = (
            self.output_size *
            self.dtype_bytes
        )
        return (
            a_bytes +
            b_bytes +
            c_bytes
        )

    @property
    def memory_mb(self):
        return (
            self.memory_bytes /
            (1024**2)
        )

    def __post_init__(self):

        if not isinstance(self.a_shape, tuple):
            raise TypeError(
                f"a_shape must be tuple, got {self.a_shape}"
            )

        if not isinstance(self.b_shape, tuple):
            raise TypeError(
                f"b_shape must be tuple, got {self.b_shape}"
            )


# =========================================================
# MEMORY
# =========================================================

def tensor_memory(
    shape,
    dtype_bytes=4
):
    return math.prod(shape) * dtype_bytes


def case_memory(
    case: Case,
    dtype_bytes=4
):
    a = tensor_memory(
        case.a_shape,
        dtype_bytes
    )
    b = tensor_memory(
        case.b_shape,
        dtype_bytes
    )
    c = tensor_memory(
        case.output_shape,
        dtype_bytes
    )
    return {
        "a_bytes": a,
        "b_bytes": b,
        "c_bytes": c,
        "total_bytes": a+b+c,
    }


# =========================================================
# FLOPS
# =========================================================

def contraction_flops(case: Case):
    reduction_size = math.prod(
        case.a_shape[i]
        for i, _ in case.cis
    )
    output_size = math.prod(
        case.output_shape
    )
    multiplies = (
        output_size *
        reduction_size
    )
    return 2 * multiplies



# =========================================================
# CASE CREATION
# =========================================================

def make_case(
    name,
    a_shape,
    b_shape,
    cis
):
    return Case(
        name=name,
        a_shape=tuple(a_shape),
        b_shape=tuple(b_shape),
        cis=tuple(cis),
    )


# =========================================================
# GENERATION
# =========================================================

def make_samples():
    cases = []

    for size_set, rank_set in zip(SIZES, RANKS):

        for rank in rank_set:
            for n in size_set:

                a = (n,) * rank
                b = (n,) * rank

                if rank < 5:
                    cases.append(
                        make_case(
                            f"{rank}r{n}s1o",
                            a,
                            b,
                            ((0,0),)
                        )
                    )

                if 3 < rank < 7:
                    order = rank // 2
                    cases.append(
                        make_case(
                            f"{rank}r{n}s{order}o",
                            a,
                            b,
                            tuple((i,i) for i in range(order))
                        )
                    )

                if rank > 5:
                    order = rank - 1
                    cases.append(
                        make_case(
                            f"{rank}r{n}s{order}o",
                            a,
                            b,
                            tuple((i,i) for i in range(order))
                        )
                    )

    rect_cases = [
        ("rect0", (64,32,16), (16,32,64), ((2,0),)),
        ("rect1", (128,32,8), (8,32,128), ((2,0),)),
        ("rect2", (256,64,16), (16,64,256), ((2,0),)),
        ("rect3", (32,64,128,16), (16,128,64,32), ((3,0),(2,1))),
    ]

    for name, a, b, cis in rect_cases:
        cases.append(make_case(name, a, b, cis))

    return cases


# =========================================================
# FILTER
# =========================================================

def filter_by_size(case):
    return case.flops > MAX_FLOPS

def filter_by_flops(accepted_flops, case):
    if case.memory_bytes > MAX_MEMORY:
        return True

    if not accepted_flops:
        return False

    lflops = math.log10(case.flops)
    nearest = min(max(lflops, 1) - max(math.log10(f), 1) for f in accepted_flops)
    if nearest < FLOPS_FILTER_DISTANCE:
        return random.random() < FLOPS_FILTER_REJECTION_PROBABILITY
    return False


def filter_cases(cases):
    unique = set()
    result = []
    accepted_flops = []

    for case in cases:
        if filter_by_flops(accepted_flops, case):
            header = "Rejected for flops: "
        elif filter_by_size(case):
            header = "Rejected for size: "
        elif case.name in unique:
            header = "Repeated: "
        else:
            unique.add(case.name)
            result.append(case)
            accepted_flops.append(case.flops)
            header = "Accepted: "
        print(header,
            case.name,
            case.a_shape,
            case.b_shape,
            case.cis,
            {
                "memory MB": case.memory_mb,
                "flops": case.flops,
            }
        )
    return result


# =========================================================
# RUN
# =========================================================

samples = sorted(filter_cases(
    make_samples()
), key=lambda x: x.name)
print(f"{len(samples)} cases")
print(f"max memory: {max(s.memory_bytes for s in samples) / 1024**2:.3e} MB")
print(f"max flops:  {max(s.flops for s in samples):.3e} FLOPS")
