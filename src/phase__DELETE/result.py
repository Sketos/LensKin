import autofit as af


class Result(af.Result):
    def __init__(
        self,
        instance,
        likelihood,
        previous_model,
        gaussian_tuples,
        analysis,
        output
    ):
        super(Result, self).__init__(
            instance=instance,
            likelihood=likelihood,
            previous_model=previous_model,
            gaussian_tuples=gaussian_tuples
        )

        self.analysis = analysis
        self.output = output

    # @property
    # def most_likely_model_data(self):
    #     return self.analysis.model_data_from_instance(
    #         instance=self.instance
    #     )
    #
    # @property
    # def most_likely_fit(self):
    #     return self.analysis.fit_from_model_data(
    #         model_data=self.most_likely_model_data
    #     )
