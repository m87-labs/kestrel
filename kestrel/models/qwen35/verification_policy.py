"""Choose a causal prefix from draft confidence and measured execution costs."""
import math
import statistics
from collections import deque
from itertools import accumulate
from operator import mul


class VerificationWidthPolicy:
    """No fitted thresholds: maximize expected committed tokens per millisecond."""
    widths = (2, 3, 4, 5, 6, 7, 8, 16)

    def __init__(self, widths=None):
        if widths is not None:
            widths = tuple(widths)
            if (not widths or any(type(width) is not int or not 2 <= width <= 16
                                  for width in widths)
                    or tuple(sorted(set(widths))) != widths):
                raise ValueError('verification widths must be unique increasing spans within the draft')
            self.widths = widths
        self.cycle_overhead_samples = deque(maxlen=16)
        self.target_samples = {}
        self.draft_samples = deque(maxlen=16)

    def reset_request(self):
        self.cycle_overhead_samples.clear()

    def record_cycle(self, width, elapsed_ms):
        if width not in self.widths or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError('valid width and positive cycle execution time required')
        if not self.calibrated or not self.draft_samples:
            return
        # Include staging, commit and host scheduling omitted by graph-only costs.
        overhead = elapsed_ms - self.draft_samples[-1] - statistics.median(self.target_samples[width])
        self.cycle_overhead_samples.append(max(0.0, overhead))

    @property
    def calibrated(self):
        return set(self.target_samples) == set(self.widths)

    def set_target_samples(self, samples):
        if (set(samples) != set(self.widths)
                or any(len(values) < 5 or any(type(x) not in (int, float)
                       or not math.isfinite(x) or x <= 0 for x in values)
                       for values in samples.values())):
            raise ValueError('complete positive verification measurements required')
        self.target_samples = {width: tuple(values) for width, values in samples.items()}

    def record_draft(self, elapsed_ms):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError('positive draft execution time required')
        self.draft_samples.append(elapsed_ms)

    def choose(self, confidence, *, sequences=1):
        if (type(sequences) is not int or sequences < 1 or len(confidence) % sequences
                or not self.widths[-1] - 1 <= len(confidence) // sequences <= 15):
            raise ValueError('confidence must cover every complete draft block')
        if not self.calibrated or not self.draft_samples:
            return self.widths[-1]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence):
            return self.widths[0]
        stride = len(confidence) // sequences
        survival = [tuple(accumulate(confidence[row * stride:(row + 1) * stride], mul))
                    for row in range(sequences)]
        draft_cost = statistics.median(self.draft_samples)
        overhead = statistics.median(self.cycle_overhead_samples) if self.cycle_overhead_samples else 0.0
        scores = {width: sum(1 + sum(prefix[:width-1]) for prefix in survival) /
                  (statistics.median(samples) + draft_cost + overhead)
                  for width, samples in self.target_samples.items()}
        return max(self.widths, key=lambda width: scores[width])
