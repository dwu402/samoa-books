""" Pypei analysis of Samoan measles outbreak of 2019 """

import pypei
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy import interpolate, optimize, stats

from plot_utils import *

##### Setup
set_font()

##### Reading in data
raw_data = pd.read_csv("../../samoa/samoa-data/samoa_press_release_data.csv", 
                       header=0, dayfirst=True, parse_dates=True, skiprows=[1])

time, cE, H, cH, D, G = raw_data.iloc[:,1:].values.T

data_nonclipped = np.vstack([H, G, D, cE, cH]).astype(float)

the_dates = list(map(lambda t: datetime.datetime.strptime(t, "%d/%m/%Y"), raw_data.Date.values))

def time_conv(t: float):
    return the_dates[0] + datetime.timedelta(days=t)

def time_conv_arr(time_arr):
    return np.array([time_conv(x) for x in time_arr])

# Defining model
def seir_model(t, y, p):
    b, g, e, d, a, mH = p[:6]
    S,E,I,H,G,R,D,cE,cH = y[:9]
    return [
        -b*S*I/(S+E+I+R+G+H),
        b*S*I/(S+E+I+R+G+H)-g*E,
        g*E-(e+a)*I,
        e*I-(d+mH)*H,
        d*H,
        a*I,
        mH*H,
        g*E,
        e*I
    ]

model_form = {
    'state': 9, 
    'parameters': 6,
}

##### Constructing problem
problem = pypei.Problem()

problem.build_model(seir_model, model_form, time_span=[time[0], time[-1]*1.25])

clip = 20
y0_ixs = [3, 4, 6, 7, 8]
problem.build_data(*problem.slice_data(time, data_nonclipped, clip=clip), iy=y0_ixs)

gamma_reg = [
    {
        'sz': (1, 1),
        'obs_fn': problem.model.ps[1], # add regularisation: gamma. Treats the small E problem.
    },
    {'n': 1, 'iid': True},
]
beta_reg = [
    {
        'sz': (1, 1),
        'obs_fn': problem.model.ps[0], # add regularisation: beta. Treats the small I problem.
    },
    {'n': 1, 'iid': True},
]
problem.build_objective({'order': ['SCR', 'EI', 'H', 'G', 'A', 'D'], 'inherent_order':'SEIHGRDCA'}, False, gamma_reg, beta_reg)

problem.build_solver(solver_opts={'ipopt': {'max_iter': 5000, 'print_frequency_iter': 100}}, guess_opts={'x0': 50_000, 'p0': 1.0}, constraint_opts={'ubg': 120_000}, 
                     w0=[1,1,1,1,1, 0.9,0.9,0.9,0.9,0.9,0.9,], weight_bounds=[[1e-1, None]]*5 + [[None, 3]]*6)
# custom weighting function for the regularisation terms
_rho = 16

def zero_p(w, y):
    return [*w, _rho, _rho, *y, 0, 0, 0]
problem.p = zero_p

# solve for MLE
solution = problem.solve(nit=6)

##### Profiling

# choose an iteration to stop at, plot MLE
mle_idx = 2
problem.plot_solution(solution, it=mle_idx, data=True)

def construct_profiler(quantity):
    config = {
        'g+': quantity,
        'pidx': ca.Function(f'pidx_{quantity.name()}', [problem.solver.decision_vars], [quantity])
    }

    profiler = pypei.fitter.Profiler(problem.solver, config)

    return profiler

# profile over Rc
R0eff_repr = problem.model.ps[0] / (problem.model.ps[2] + problem.model.ps[4]) * (problem.model.xs[0,0]/ca.sum2(problem.model.xs[0,:-2]))

R0profiler = construct_profiler(R0eff_repr)
mle_r0 = float(R0profiler.p_locator(solution['shist'][mle_idx]['x']))
r0pbounds = R0profiler.symmetric_bound_sets(solution['shist'][mle_idx], num=21, variance=0.1)

r0profiles, = problem.do_profile(R0profiler, solution['shist'][mle_idx], solution['whist'][mle_idx], [r0pbounds], repair=True)

# profile over total cases
tc_var = problem.model.xs[-1, 7]

tcprofiler = construct_profiler(tc_var)
mle_tc = float(tcprofiler.p_locator(solution['shist'][mle_idx]['x']))
tcpbounds = tcprofiler.symmetric_bound_sets(solution['shist'][mle_idx], num=21, variance=0.1)

tcprofiles, = problem.do_profile(tcprofiler, solution['shist'][mle_idx], solution['whist'][mle_idx], [tcpbounds], repair=True)

# profile over total deaths
mort_var = problem.model.xs[-1, 6]

mortprofiler = construct_profiler(mort_var)
mle_mort = float(mortprofiler.p_locator(solution['shist'][mle_idx]['x']))
mortpbounds = mortprofiler.symmetric_bound_sets(solution['shist'][mle_idx], num=21, variance=0.2)

mortprofiles, = problem.do_profile(mortprofiler, solution['shist'][mle_idx], solution['whist'][mle_idx], [mortpbounds], repair=True)


##### construct profile normalised likelihoods
# define the likelihood function over which to normalise the profile
ll_fn = ca.Function('llfn', [problem.solver.decision_vars, problem.solver.parameters], [problem.objective.unweighted_log_likelihood])
def ll_direct(sol):
    return ll_fn(sol['s']['x'] , problem.p(solution['whist'][mle_idx], problem.data))

ninefive = np.exp(-1/2 * (stats.chi2.ppf(q=0.95, df=1)))

def construct_normalised_likelihood(profiles):
    prof_f = np.array([ll_direct(s) for s in profiles.values()]).flatten()
    return np.exp(-0.5 * (prof_f - min(prof_f)))

norm_r0_prof = construct_normalised_likelihood(r0profiles)
norm_tc_prof = construct_normalised_likelihood(tcprofiles)
norm_mort_prof = construct_normalised_likelihood(mortprofiles)

# approximate with splines for root finding
def spline_approx(profiles, normalised_likelihood):
    x_vals = profiles.keys()
    y_vals = normalised_likelihood

    return interpolate.interp1d(x_vals, y_vals, kind='cubic', bounds_error=False, fill_value=0.0)

r0_spl = spline_approx(r0profiles, norm_r0_prof)
tc_spl = spline_approx(tcprofiles, norm_tc_prof)
mort_spl = spline_approx(mortprofiles, norm_mort_prof)

# compute 95% confidence intervals via root finding
def profile_ci(profiles, mle_val, spl, q=0.95):
    threshold = np.exp(-1/2 * (stats.chi2.ppf(q=q, df=1)))
    xlo = optimize.brentq(lambda x: spl(x) - threshold, min(profiles.keys()), mle_val)
    xhi = optimize.brentq(lambda x: spl(x) - threshold, mle_val, max(profiles.keys()))

    return [xlo, xhi]

r0_profile_ci = profile_ci(r0profiles, mle_r0, r0_spl)
tc_profile_ci = profile_ci(tcprofiles, mle_tc, tc_spl)
mort_profile_ci = profile_ci(mortprofiles, mle_mort, mort_spl)

##### Bivariate profile
comb_num = 21

combined_profiler = construct_profiler(ca.vcat([R0eff_repr, tc_var]))
mle_comb = combined_profiler.p_locator(solution['shist'][mle_idx]['x']).toarray().flatten()
comb_bound_sets = combined_profiler.symmetric_nvariate_bound_sets(solution['shist'][mle_idx], num=comb_num, variance=0.1)

bivar_profiles, = problem.do_profile(combined_profiler, solution['shist'][mle_idx], solution['whist'][mle_idx], comb_bound_sets)

r0_mesh, tc_mesh = [np.array(x).reshape((comb_num, comb_num)) for x in zip(*bivar_profiles.keys())]
norm_bivar_prof = construct_normalised_likelihood(bivar_profiles).reshape((comb_num, comb_num))

##### RML setup

pypei.fitter.reconfig_rto(problem.model, problem.objective, problem.solver, problem.objective_config, index=1)

mle_w_full = problem.p(solution['whist'][mle_idx], [])[:-3]
# generate RML samples
rys = problem.solver._generate_gaussian_samples(solution['shist'][mle_idx], [problem.data, 0, 0, 0], mle_w_full, problem.objective, 200)

def rs_p(w, sample):
    return [*w, _rho, _rho, *[i for s in sample for i in s]]

# fit RMl samples, reject non convergence
rss = problem.solver._fit_samples(rys, solution['shist'][mle_idx]['x'], rs_p, solution['whist'][mle_idx], weight=problem.weight_fn, weight_args=problem.weight_args, nit=1, lbg=problem.lbg, ubg=problem.ubg, lbx=problem.lbx, ubx=problem.ubx, must_converge=True)

raw_rml_states = [problem.solver.get_state(rx[0], problem.model) for rx in rss]

rml_r0 = np.array([R0profiler.p_locator(s[0]['x']) for s in rss]).flatten()
rml_tc = np.array([tcprofiler.p_locator(s[0]['x']) for s in rss]).flatten()
rml_mort = np.array([mortprofiler.p_locator(s[0]['x']) for s in rss]).flatten()

# plot RML time series of total cases
plt.figure(figsize=[6.5, 4.5])

for s, _ in rss:
    plt.plot(time_conv_arr(problem.model.observation_times), problem.solver.get_state(s, problem.model)[:,7], 'k', alpha=0.5, linewidth=0.15)
for yy in rys[0]:
    nangrid = np.nan * np.ones(problem.data_indexer.shape)
    nangrid[problem.data_indexer] = yy

plt.plot(the_dates[:clip], cE[:clip], 'o', color='r', label='Data (Fitted)', mfc='none')
plt.plot(the_dates[clip:], cE[clip:], 'x', color='r', label='Data (Unfitted)', mfc='none')
plt.axvline(the_dates[clip-1], color='red', linestyle='dashdot')
plt.plot(time_conv_arr(problem.model.observation_times), problem.solver.get_state(solution['shist'][mle_idx], problem.model)[:,7], '--', color='dodgerblue', linewidth=2, label='MLE')
plt.legend(prop=legend_font)
plt.xlabel('Date', fontproperties=label_font)
plt.ylabel("Cumulative Cases", fontproperties=label_font)
form_xmonths(plt.gca())
plt.xticks(fontproperties=tick_font)
plt.yticks(fontproperties=tick_font)

##### Plot combined RML and profile likelihoods

def plot_confidence(pl_pnts, pl_rng, pl_spl, pl_ci, rml_samples, xlabel, rml_loc, truth=None):
    f = plt.figure(figsize=[8,4.5])
    ax = f.add_subplot()
    
    # plot rml_Samples
    ax.hist(rml_samples, bins=21, color='dodgerblue', label="RML Samples")
    rml_ci = [np.quantile(rml_samples, 0.025), np.quantile(rml_samples, 0.975)]
    for x in rml_ci:
        ax.plot([x, x], [0.75*rml_loc, 1.25*rml_loc], color='k')
    ax.plot(rml_ci, [rml_loc, rml_loc], color='k', label="Bootstrap Interval")
    
    ax2=ax.twinx()
    
    ax2.plot(pl_rng, pl_spl(pl_rng), 'k',)# label="Profile")
    ax2.plot(*pl_pnts, 'ko')
    
    if truth is not None:
        ax.axvline(truth, color='r', linestyle='dotted', label='Truth')
    
    ax.set_xlabel(xlabel, fontproperties=label_font)
    ax2.set_xlabel(xlabel, fontproperties=label_font)
    ax.set_ylabel("RML Sample Frequency", fontproperties=label_font)
    ax2.set_ylabel("Profile Likelihood", fontproperties=label_font)

    for l in ax.get_yticklabels():
        l.set_fontproperties(tick_font)
    for l in ax2.get_yticklabels():
        l.set_fontproperties(tick_font)
    for l in ax.get_xticklabels():
        l.set_fontproperties(tick_font)
    for l in ax2.get_xticklabels():
        l.set_fontproperties(tick_font)

    ninefive = np.exp(-1/2 * (stats.chi2.ppf(q=0.95, df=1)))
    ax2.axhline(ninefive, color='k', linestyle='--', label="Profile Interval")
    for ci in pl_ci:
        ax2.axvline(ci, color='k', linestyle='--')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    
    handles_ax, _ = ax.get_legend_handles_labels()
    handles_ax2, _ = ax2.get_legend_handles_labels()

    handles = [*handles_ax2, *handles_ax]
    handles.insert(1, lines.Line2D([0], [0], color='k', marker='o', label='Profile Likelihood'))

    f.legend(handles=handles, loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, framealpha=1.0, prop=legend_font)

def make_range(profiles):
    return np.linspace(min(profiles.keys()), max(profiles.keys()), 1001)

def mk_pnts(profiles, norm_ll):
    return [profiles.keys(), norm_ll]

plot_confidence(mk_pnts(r0profiles, norm_r0_prof), make_range(r0profiles), r0_spl, r0_profile_ci, rml_r0, r"$\mathcal{R}_c$", 3)

plot_confidence(mk_pnts(tcprofiles, norm_tc_prof), make_range(tcprofiles), tc_spl, tc_profile_ci, rml_tc, r"Total Cases", 4, truth=cE[-1])

plot_confidence(mk_pnts(mortprofiles, norm_mort_prof), make_range(mortprofiles), mort_spl, mort_profile_ci, rml_mort, r"Total Deaths", 4, truth=D[-1])

##### Plot bivariate likelihood
plt.figure(figsize=[9, 5])
plt.contourf(tc_mesh, r0_mesh, norm_bivar_prof, np.linspace(0, 1, 21), cmap='plasma')
ylims = plt.ylim()
xlims = plt.xlim()
cbar = plt.colorbar()
plt.contour(tc_mesh, r0_mesh, norm_bivar_prof, [ninefive], cmap='gray')

R_over_tc = [float(R0profiler.p_locator(s['s']['x'])) for s in tcprofiles.values()]
tc_line, = plt.plot(list(tcprofiles.keys()), R_over_tc, label='Total Cases Profile', color='dodgerblue', linewidth=4)
rc_line, = plt.plot([np.squeeze(problem.solver.get_state(s['s'], problem.model)[-1, -2]) for s in r0profiles.values()], r0profiles.keys(), label="$R_c$ Profile", color='green', linewidth=4)
    
plt.scatter(rml_tc, rml_r0, edgecolors='k', color='white', label='RML Samples', marker='.', zorder=4)
    
plt.xlabel('Total Cases', fontproperties=label_font)
plt.ylabel('$R_c$', fontproperties=label_font)
cbar.set_label(label='Normalised Likelihood', fontproperties=label_font)
cbar.ax.tick_params(labelsize=tick_font.get_size())
plt.axhline(mle_r0, color='white', linestyle='dashed', label='MLE')
plt.axvline(mle_tc, color='white', linestyle='dashed')
plt.axvline(cE[-1], color='red', linestyle='dashdot', label='True Total Cases')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(fontproperties=tick_font)
plt.yticks(fontproperties=tick_font)
plt.legend(prop=legend_font, loc='lower left')

plt.show()