import numpy as np
import neo
import quantities as pq
from meegkit.dss import dss0

class JD():
    def __init__(self, n_max_components=None, explained_variance_threshold=1e-9):
        """
        n_max_components: int
            Number of PCs to retain (default=all).
        explained_variance_threshold: float
            Ignore PCs smaller than explained_variance_threshold (default=10.^-9).
        """
        self.n_max_components = n_max_components
        self.explained_variance_threshold = explained_variance_threshold

    def fit(self, baseline_data=None, bias_filtered_data=None,
            baseline_covmat=None, bias_filtered_covmat=None, verbose=False):

        # check data shape compatibility
        if baseline_covmat is None and bias_filtered_covmat is None:
            if baseline_data.shape[0] != bias_filtered_data.shape[0]:
                raise ValueError('`baseline_data` and `bias_filtered_data`'
                                 'should have same size in the first dimension.')
        else:
            if not (baseline_covmat.shape[0] == baseline_covmat.T.shape[1] and
                    np.all(np.ravel(baseline_covmat) == np.ravel(baseline_covmat.T))):
                raise ValueError('`baseline_covmat` should be a symmetric square'
                                 'matrix.')
            if not (bias_filtered_covmat.shape[0] == bias_filtered_covmat.T.shape[1] and
                    np.all(np.ravel(bias_filtered_covmat) == np.ravel(bias_filtered_covmat.T))):
                raise ValueError('`bias_filtered_covmat` should be a symmetric square'
                                 'matrix.')
            # check data shape compatibility
            if baseline_covmat.shape != bias_filtered_covmat.shape:
                raise ValueError('`baseline_covmat` and `bias_filtered_covmat` should have same'
                                 'size.')

        if baseline_covmat is None and bias_filtered_covmat is None:
            c0 = np.cov(baseline_data)
            c1 = np.cov(bias_filtered_data)
        else:
            c0 = baseline_covmat
            c1 = bias_filtered_covmat

        return self._fit(c0, c1, verbose)

    def _fit(self, c0, c1, verbose):

        todss, fromdss, pwr0, pwr1 = dss0(c0, c1, keep1=self.n_max_components,
                                          keep2=self.explained_variance_threshold)

        self.todss = todss
        self.n_components = self.todss.shape[1]
        self.fromdss = fromdss
        self.pwr0 = pwr0
        self.pwr1 = pwr1
        self.power_ratio = self.pwr1/self.pwr0

        return self

    def transform(self, data):

        if data.shape[0] != self.todss.shape[0]:
            raise ValueError('The first dimension of `data` should be equal to the number of channels.')

        return np.dot(data.T, self.todss).T

    def fit_transform(self, baseline_data=None, bias_filtered_data=None,
            baseline_covmat=None, bias_filtered_covmat=None, verbose=False):
        self.fit(baseline_data=baseline_data,
                 bias_filtered_data=bias_filtered_data,
                 baseline_covmat=baseline_covmat,
                 bias_filtered_covmat=bias_filtered_covmat,
                 verbose=verbose)
        return self.transform(baseline_data)

    def project_out(self, data, components_to_discard=[],
                    power_ratio_threshold=None):
        """
        project_out here means to project from dss space to original space leaving specific components out of the data

        components_to_discard should be a list of indices of component not to use for the projection
        """

        if data.shape[0] != self.todss.shape[0]:
            raise ValueError('The first dimension of `data` should be equal to the number of channels.')
        if not isinstance(components_to_discard, list):
            raise ValueError('`components_to_discard` should be a list.')
        if len(components_to_discard) > self.n_components:
            raise ValueError('`components_to_discard` should be empty or a subset of all components.')
        if len(components_to_discard) != 0 and max(components_to_discard) >= self.n_components:
            raise ValueError('The indices in `components_to_discard` cannot exceed {} (`n_components`)'.format(self.n_components))

        if len(components_to_discard) != 0 and power_ratio_threshold != None:
            raise ValueError('Either one of `components_to_discard` or `power_ratio_threshold` has to be specified.')

        if len(components_to_discard) == 0:
            components_to_discard = np.where(self.power_ratio > power_ratio_threshold)[0]

        mask = np.ones(self.n_components, dtype=bool)
        mask[components_to_discard] = False
        return self._project(data, mask)

    def _project(self, data, mask):
        if np.all(mask) == True:
            return data
        e_diag = np.zeros(self.n_components)
        e_diag[mask] = 1
        e = np.diag(e_diag)
        return np.dot(np.dot(np.dot(data.T, self.todss), e), self.fromdss).T

    def project_in(self, data, components_to_keep=[],
                   power_ratio_threshold=None):
        """
        project_in here means to project from dss space to original space using specific components in the data

        components_to_keep should be a list of indices of component not to use for the projection
        """

        if data.shape[0] != self.todss.shape[0]:
            raise ValueError('The first dimension of `data` should be equal to the number of channels.')
        if not isinstance(components_to_keep, list):
            raise ValueError('`components_to_keep` should be a list.')
        if len(components_to_keep) > self.n_components:
            raise ValueError('`components_to_keep` should be empty or a subset of all components.')
        if len(components_to_keep) != 0 and max(components_to_keep) >= self.n_components:
            raise ValueError('The indices in `components_to_keep` cannot exceed {} (`n_components`)'.format(self.n_components))


        if len(components_to_keep) != 0 and power_ratio_threshold != None:
            raise ValueError('Either one of `components_to_keep` or `power_ratio_threshold` has to be specified.')

        if len(components_to_keep) == 0:
            components_to_keep = np.where(self.power_ratio > power_ratio_threshold)[0]

        mask = np.zeros(self.n_components, dtype=bool)
        mask[components_to_keep] = True
        return self._project(data, mask)
