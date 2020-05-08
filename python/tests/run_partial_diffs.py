from util.central_differences import central_differences
from util.sample import sample_sympd
import numpy as np

""" Test for the partial derivatives in section 2.1.1 of the document"""


np.random.seed(0)

dim = 10

q_old = np.random.normal(size=dim)
Q_old = sample_sympd(dim)


def logdet(Q_flat):
    q = np.reshape(Q_flat, [dim, dim])
    return np.linalg.slogdet(q)[1]


def trace_term(Q_flat):
    Q = np.reshape(Q_flat, [dim, dim])
    return np.trace(Q_old @ np.linalg.inv(Q))


def m1_term(q, Q_flat):
    Q = np.reshape(Q_flat, [dim, dim])
    return q @ np.linalg.solve(Q, q_old)


def m2_term(q, Q_flat):
    Q_inv = np.linalg.inv(np.reshape(Q_flat, [dim, dim]))
    return np.dot(q, Q_inv @ Q_old @ Q_inv @ q)


q = np.random.normal(size=dim)
Q = sample_sympd(dim)

"""log det term"""
print("Logdet", end=" ")
grad_numeric = central_differences(logdet, np.reshape(Q, -1))
grad_analytic = np.linalg.inv(Q)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

"""trace """
print("Trace", end=" ")
grad_numeric = central_differences(trace_term, np.reshape(Q, -1))
grad_analytic = - np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

"""m1_term """
print("M1_term, q", end=" ")
grad_numeric = central_differences(lambda x: m1_term(x, np.reshape(Q, -1)), q)
grad_analytic = np.linalg.inv(Q) @ q_old
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("M1_term, Q", end=" ")
grad_numeric = central_differences(lambda x: m1_term(q, x), np.reshape(Q, -1))
grad_analytic = - np.linalg.inv(Q) @ np.outer(q, q_old) @ np.linalg.inv(Q)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

"""m2_term """
print("M2_term, q", end=" ")
grad_numeric = central_differences(lambda x: m2_term(x, np.reshape(Q, -1)), q)
grad_analytic = 2 * np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q) @ q
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("M2_term, Q", end=" ")
grad_numeric = central_differences(lambda x: m2_term(q, x), np.reshape(Q, -1))
ga = Q_old @ np.linalg.inv(Q) @ np.outer(q, q)
grad_analytic = - np.linalg.inv(Q) @ (ga + ga.T) @ np.linalg.inv(Q)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print("------------ SECOND ----------------")
q_target = np.random.normal(size=dim)
Q_target = sample_sympd(dim)

eta = 12.2456
omega = 7.64764


def get_q(eta_, omega_, q_target_):
    return (eta_ * q_old + q_target_) / (eta_ + omega_ + 1)


def get_Q(eta_, omega_, Q_target_):
    return (eta_ * Q_old + Q_target_) / (eta_ + omega_ + 1)


q = get_q(eta, omega, q_target)
Q = get_Q(eta, omega, Q_target)

print("d_q d_eta", end=" ")
grad_numeric = central_differences(lambda x: get_q(x, omega, q_target), eta, dim=dim)
grad_analytic = ((omega + 1) * q_old - q_target) / ((omega + eta + 1)**2)
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("d_Q d_eta", end=" ")
grad_numeric = central_differences(lambda x: get_Q(x, omega, Q_target), eta, dim=[dim, dim])
grad_analytic = ((omega + 1) * Q_old - Q_target) / ((omega + eta + 1)**2)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_q d_omega", end=" ")
grad_numeric = central_differences(lambda x: get_q(eta, x, q_target), omega, dim=dim)
grad_analytic = - (eta * q_old + q_target) / ((omega + eta + 1)**2)
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("d_Q d_omega", end=" ")
grad_numeric = central_differences(lambda x: get_Q(eta, x, Q_target), omega, dim=[dim, dim])
grad_analytic = - (eta * Q_old + Q_target) / ((omega + eta + 1)**2)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_q d_q_tilde", end=" ")
grad_numeric = central_differences(lambda x: get_q(eta, omega, x), q_target, dim=dim)
grad_analytic = np.eye(dim) / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_tm1 d_q_tilde", end=" ")
grad_numeric = central_differences(lambda x: m1_term(get_q(eta, omega, x), np.reshape(Q, [-1])), q_target)
                        # d_q d_q_ilde              # d_tm1 d_q
grad_analytic = np.linalg.inv(Q) @ q_old / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("d_tm2 d_q_tilde", end=" ")
grad_numeric = central_differences(lambda x: m2_term(get_q(eta, omega, x), np.reshape(Q, [-1])), q_target)
                        # d_q d_q_ilde                          # 0.5 * d_tm2 d_q
grad_analytic = 2 * np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q) @ q / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim]) - grad_analytic)))

print("d_tlogdet d_Q_tilde", end=" ")
grad_numeric = central_differences(lambda x: logdet(get_Q(eta, omega, np.reshape(x, [dim, dim]))), np.reshape(Q_target, [-1]))
grad_analytic = np.linalg.inv(Q) / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_ttrace d_Q_tilde", end=" ")
grad_numeric = central_differences(lambda x: trace_term(get_Q(eta, omega, np.reshape(x, [dim, dim]))), np.reshape(Q_target, [-1]))
grad_analytic = - np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q) / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_tm1 d_Q_tilde", end=" ")
grad_numeric = central_differences(lambda x: m1_term(q, get_Q(eta, omega, np.reshape(x, [dim, dim]))), np.reshape(Q_target, [-1]))
grad_analytic = - np.linalg.inv(Q) @ np.outer(q, q_old) @ np.linalg.inv(Q) / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_tm2 d_Q_tilde", end=" ")
grad_numeric = central_differences(lambda x: m2_term(q, get_Q(eta, omega, np.reshape(x, [dim, dim]))), np.reshape(Q_target, [-1]))
tmp = Q_old @ np.linalg.inv(Q) @ np.outer(q, q)
grad_analytic = - np.linalg.inv(Q) @ (tmp + tmp.T) @ np.linalg.inv(Q) / (omega + eta + 1)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))


d_q_d_eta = ((omega + 1) * q_old - q_target) / ((omega + eta + 1)**2)
d_Q_d_eta = ((omega + 1) * Q_old - Q_target) / ((omega + eta + 1)**2)

print("d_tlogdet d_eta", end=" ")
grad_numeric = central_differences(lambda x: logdet(np.reshape(get_Q(x, omega, Q_target), [-1])), eta)
grad_analytic = np.trace(np.linalg.inv(Q) @ d_Q_d_eta)
print(np.max(np.abs(grad_numeric[0, 0] - grad_analytic)))

print("d_trace d_eta", end=" ")
grad_numeric = central_differences(lambda x: trace_term(np.reshape(get_Q(x, omega, Q_target), [-1])), eta)
grad_analytic = np.trace(- np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q) @ d_Q_d_eta)
print(np.max(np.abs(grad_numeric[0, 0] - grad_analytic)))

print("d_tm1 d_eta", end=" ")
grad_numeric = central_differences(lambda x: m1_term(get_q(x, omega, q_target), np.reshape(get_Q(x, omega, Q_target), [-1])), eta)
grad_analytic = np.dot(np.linalg.inv(Q) @ q_old, d_q_d_eta) + \
                np.trace(- np.linalg.inv(Q) @ np.outer(q, q_old) @ np.linalg.inv(Q) @ d_Q_d_eta)
print(np.max(np.abs(grad_numeric[0, 0] - grad_analytic)))

print("d_tm2 d_eta", end=" ")
grad_numeric = central_differences(lambda x:  m2_term(get_q(x, omega, q_target), np.reshape(get_Q(x, omega, Q_target), [-1])), eta)
tmp = Q_old @ np.linalg.inv(Q) @ np.outer(q, q)
grad_analytic = np.dot(2 * np.linalg.inv(Q) @ Q_old @ np.linalg.inv(Q)  @ q, d_q_d_eta) + \
                np.trace(- np.linalg.inv(Q) @ (tmp + tmp.T) @ np.linalg.inv(Q) @ d_Q_d_eta)
print(np.max(np.abs(grad_numeric[0, 0] - grad_analytic)))


