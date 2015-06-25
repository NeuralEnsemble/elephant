import numpy as np
from scipy import interpolate

def check_for_duplicated_electrodes(elec_pos):
    """
    **Parameters**
    elec_pos : np.array
    **Returns**
    has_duplicated_elec : Boolean
    """
    unique_elec_pos = np.vstack({tuple(row) for row in elec_pos})
    has_duplicated_elec = unique_elec_pos.shape == elec_pos.shape
    return has_duplicated_elec

def sparse_dist_table(R, dist_max, dt_len):
    """
    **Returns**

    xs : np.array
        sparsely probed indices from the distance table
    """
    dense_step = 3
    denser_step = 1
    sparse_step = 9
    border1 = 0.9 * R/dist_max * dt_len
    border2 = 1.3 * R/dist_max * dt_len

    xs = np.arange(0, border1, dense_step)
    xs = np.append(xs, border1)
    zz = np.arange((border1 + denser_step), border2, dense_step)

    xs = np.concatenate((xs, zz))
    xs = np.append(xs, [border2, (border2 + denser_step)])
    xs = np.concatenate((xs, np.arange((border2 + denser_step +
                                        sparse_step/2), 
                                       dt_len,
                                       sparse_step)))
    xs = np.append(xs, dt_len + 1)
    xs = np.unique(np.array(xs))
    return xs

def interpolate_dist_table(xs, probed_dist_table, dt_len):
    '''Interpolates the dist tables values over the required density'''
    inter = interpolate.interp1d(x=xs, y=probed_dist_table,
                                    kind='cubic', fill_value=0.0 )
    dt_int = np.array([inter(i) for i in xrange(dt_len)])
    dt_int.flatten()
    return dt_int

def calc_error(k_pot, pots, lambd, index_generator):
    '''Useful for Cross validation - when done in parallel'''
    err = 0
    for idx_train, idx_test in index_generator:
        B_train = k_pot[np.ix_(idx_train, idx_train)]
        V_train = pots[idx_train]
        V_test = pots[idx_test]
        I = np.identity(len(idx_train))
        B_new = np.matrix(B_train) + (lambd*I)
        beta_new = np.dot(np.matrix(B_new).I, np.matrix(V_train))
        #beta_new = np.dot(faster_inverse(B_new), np.matrix(V_train))
        B_test = k_pot[np.ix_(idx_test, idx_train)]
        V_est = np.zeros((len(idx_test),1))
        for ii in range(len(idx_train)):
            V_est += beta_new[ii,0] * B_test[:, ii]
        err += np.linalg.norm(V_est-V_test)
    return err
