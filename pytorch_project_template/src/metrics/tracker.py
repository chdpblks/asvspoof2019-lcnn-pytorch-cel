import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self._metrics = {}
        self.reset()

    def add_metric(self, key: str, metric):
        """
        Add a metric object (BaseMetric instance).
        """
        self._metrics[key] = metric

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        for metric in self._metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        result = dict(self._data.average)
        
        for key, metric in self._metrics.items():
            if hasattr(metric, 'compute_final_eer'):
                final_eer = metric.compute_final_eer()
                result[key] = final_eer
                # updates average for logging
                self._data.loc[key, "average"] = final_eer
        
        return result

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
