from dataclasses import dataclass
from math import prod
from typing import Tuple, List


# =========================================================
# LIMITS
# =========================================================

MAX_FLOPS = 10**15
MAX_MEMORY = 2**28

SIZES = [
    [2, 4, 8, 16, 32, 64, 128, 256],
    [3, 7, 15, 31, 63, 127, 255],
]

RANKS = [
    [2, 3, 4, 5, 6, 7, 8, 16, 32, 64],
    [3, 4, 5, 6, 13, 19],
]


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
        return prod(self.a_shape)

    @property
    def b_size(self):
        return prod(self.b_shape)

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
        return prod(self.output_shape)

    @property
    def reduction_size(self):
        return prod(
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
    return prod(shape) * dtype_bytes


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
    reduction_size = prod(
        case.a_shape[i]
        for i, _ in case.cis
    )
    output_size = prod(
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

def filter_cases(cases):
    unique = set()
    result = []
    max_flops = 0
    max_memory = 0

    for case in cases:
        if (
            case.name not in unique
            and
            case.flops <= MAX_FLOPS
            and
            case.memory_bytes <= MAX_MEMORY
        ):
            unique.add(case.name)
            result.append(case)
            max_flops = max(
                    max_flops,
                    case.flops
                    )
            max_memory = max(
                    max_memory,
                    case.memory_bytes
                    )
            # print(
            #     case.name,
            #     case.a_shape,
            #     case.b_shape,
            #     case.cis,
            #     {
            #         "memory MB": case.memory_mb,
            #         "flops": case.flops,
            #     }
            # )
    print(f"{len(cases)} cases")
    print( f"max memory: {max_memory / 1024**3:.2f} GB")
    print( f"max flops: {max_flops:.3e}")
    return result


# =========================================================
# RUN
# =========================================================

samples = filter_cases(
    make_samples()
)
