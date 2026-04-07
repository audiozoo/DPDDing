% fig3_ding2004.m
%
% Reproduces Figure 3 from:
%   L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim, C. R. Giardina,
%   "A robust digital baseband predistorter constructed using memory polynomials,"
%   IEEE Trans. Commun., vol. 52, no. 1, pp. 159–165, Jan. 2004.
%
% Figure 3 caption (verbatim):
%   "Effectiveness of predistortion in suppressing spectral regrowth when the PA
%    is modelled by a W-H system. (a) Output without predistortion.
%    (b) Output with memoryless predistortion.
%    (c) Output with memory polynomial predistortion (Q=2, K=5).
%    (d) Original input.  (c) and (d) almost coincide."
%
% ── Paper equations implemented ──────────────────────────────────────────
%  Eq. (4)  Memory polynomial predistorter output:
%             z(n) = Σ_{k=1,3,5} Σ_{q=0}^{Q}  a_{kq} x(n-q)|x(n-q)|^{k-1}
%
%  Eq. (5)  Basis function for ILA training block A:
%             u_{kq}(n) = [y(n-q)/G] · |y(n-q)/G|^{k-1}
%
%  Eq. (6)  Matrix form:  z = U a
%             z  = [z(0),…,z(N-1)]^T   (current predistorter output on training data)
%             U  = [u_{10},…,u_{KQ}]   (N × n_coeff basis matrix)
%
%  Eq. (7)  Least-squares solution:  â = (U^H U)^{-1} U^H z
%
%  Eq. (8)  W-H LTI blocks:
%             H(z) = (1 + 0.5 z^{-2}) / (1 − 0.2 z^{-1})
%             G(z) = (1 − 0.1 z^{-2}) / (1 − 0.4 z^{-1})
%
%  Eq. (9)  Memoryless nonlinearity:
%             w(n) = Σ_{k=1,3,5} b_k v(n)|v(n)|^{k-1}
%
%  Eq. (10) Nonlinearity coefficients (extracted from a Class-AB PA):
%             b1 = 1.0108 + 0.0858j
%             b3 = 0.0879 − 0.1583j
%             b5 = −1.0992 − 0.8891j

clear; clc; close all;
rng(42);

%% ── Simulation parameters ────────────────────────────────────────────────
fs        = 20e6;    % Sample rate: places 3 WCDMA carriers at ±0.5 normalised freq
N         = 2^16;    % Total samples used for PSD estimation
N_train   = 8000;    % Training samples — stated in paper Section V
chip_rate = 3.84e6;  % WCDMA chip rate
K         = 5;       % Max polynomial order (odd terms: 1, 3, 5)   — paper Sec. V
Q_ml      = 0;       % Memory depth: memoryless predistorter
Q_mp      = 2;       % Memory depth: memory polynomial — stated in paper Sec. V / Fig. 3

%% ── W-H PA model: exact coefficients from paper ──────────────────────────

% LTI blocks — eq. (8)
H_b = [1, 0, 0.5];   H_a = [1, -0.2];    % H(z) = (1+0.5z^{-2})/(1−0.2z^{-1})
G_b = [1, 0, -0.1];  G_a = [1, -0.4];    % G(z) = (1−0.1z^{-2})/(1−0.4z^{-1})

% Memoryless nonlinearity coefficients — eq. (10)
b1 =  1.0108 + 0.0858j;
b3 =  0.0879 - 0.1583j;
b5 = -1.0992 - 0.8891j;

% Convenience handle so PA parameters need not be re-listed everywhere
pa = @(u) pa_wh(u, H_b, H_a, G_b, G_a, b1, b3, b5);

%% ── Generate 3-carrier WCDMA-like baseband signal ────────────────────────
% Three carriers at −5 MHz, 0 Hz, +5 MHz (standard 5 MHz WCDMA spacing).
% Each carrier: LPF-shaped complex Gaussian noise — same spectral statistics
% as a root-raised-cosine filtered QPSK stream at chip_rate = 3.84 Mcps.

bw_norm = chip_rate / (fs/2);        % per-carrier bandwidth, normalised to Nyquist
h_lp    = fir1(127, bw_norm/2);      % carrier-shaping lowpass FIR
n_vec   = (0:N-1).';
x = zeros(N, 1);
for fc_hz = [-5e6, 0, 5e6]
    raw = (randn(N,1) + 1j*randn(N,1)) / sqrt(2);
    x   = x + filter(h_lp, 1, raw) .* exp(1j*2*pi*(fc_hz/fs)*n_vec);
end

% Set drive level so the PA operates in its nonlinear regime.
% H(z) DC gain = 1.875 amplifies the centre carrier; targeting rms(v_centre)
% near the polynomial's 1-dB compression point (~|v|=0.6).
x = x * 0.3 / rms(x);

%% ── (a) PA output with no predistortion ──────────────────────────────────
y_nodpd = pa(x);

%% ── Predistorter training via iterative ILA (paper Sec. III–IV) ──────────
% Extract training block (8000 samples as stated in paper Section V)
x_tr = x(1:N_train);

% (b) Memoryless predistorter: K=5, Q=0 [paper: "memoryless predistortion"]
a_ml = ila_train(x_tr, pa, K, Q_ml);

% (c) Memory polynomial predistorter: K=5, Q=2 [paper Fig. 3 caption]
a_mp = ila_train(x_tr, pa, K, Q_mp);

%% ── PA outputs with DPD applied to full signal ───────────────────────────
% Predistorter output: eq. (4)  z(n) = Σ a_{kq} x(n-q)|x(n-q)|^{k-1}
y_ml = pa(mp_basis(x, K, Q_ml) * a_ml);   % (b) memoryless DPD
y_mp = pa(mp_basis(x, K, Q_mp) * a_mp);   % (c) memory poly DPD

%% ── Power spectral density (Welch) ───────────────────────────────────────
nfft     = 4096;
win      = hann(nfft);
noverlap = nfft / 2;

[P_x,  f_hz] = pwelch(x,       win, noverlap, nfft, fs, 'twosided');
[P_nd, ~    ] = pwelch(y_nodpd, win, noverlap, nfft, fs, 'twosided');
[P_ml, ~    ] = pwelch(y_ml,    win, noverlap, nfft, fs, 'twosided');
[P_mp, ~    ] = pwelch(y_mp,    win, noverlap, nfft, fs, 'twosided');

% Shift to centred axis; normalise to ±1 (paper x-axis convention)
f_c    = f_hz - fs * (f_hz >= fs/2);
f_norm = f_c / (fs/2);
[f_norm, sidx] = sort(f_norm);

% Sort all PSDs to the centred axis
P_x_s  = P_x(sidx);
P_nd_s = P_nd(sidx);
P_ml_s = P_ml(sidx);
P_mp_s = P_mp(sidx);

% Normalise each curve to its own in-band peak so all four passbands
% align at 0 dB — matches the paper where all curves share the same
% passband level. The PA has gain G≠1, so a single shared reference would
% offset the input curve (d) vs the PA output curves (a)–(c).
inband = abs(f_norm) < 0.75;                    % mask covering all 3 carriers
dBn    = @(Ps) 10*log10(Ps / max(Ps(inband)));  % normalise to in-band peak

%% ── Figure ───────────────────────────────────────────────────────────────
figure('Color', 'w', 'Position', [100 100 680 520]);

% Plot order matches paper caption: (a) worst → (d) reference
plot(f_norm, dBn(P_nd_s), 'r-',                'LineWidth', 1.4); hold on;
plot(f_norm, dBn(P_ml_s), 'b--',               'LineWidth', 1.4);
plot(f_norm, dBn(P_mp_s), 'Color',[0 0.55 0],  'LineStyle','-',  'LineWidth', 2.0);
plot(f_norm, dBn(P_x_s),  'Color',[0.6 0 0.7], 'LineStyle','--', 'LineWidth', 1.2);

xlim([-1 1]);
ylim([-90 5]);
grid on; box on;

xlabel('Normalized Frequency',  'FontSize', 12);
ylabel('PSD (dB)',               'FontSize', 12);
title({'Effectiveness of predistortion in suppressing spectral regrowth'; ...
       '(W–H PA model)  —  Ding et al. 2004, Fig. 3'}, 'FontSize', 11);

legend({'(a) No predistortion', ...
        '(b) Memoryless predistortion', ...
        '(c) Memory poly. predistortion (Q=2, K=5)', ...
        '(d) Original input'}, ...
       'Location', 'south', 'FontSize', 10);

% =========================================================================
%  LOCAL FUNCTIONS  — must appear after all script body code in MATLAB
% =========================================================================

function a = ila_train(x_tr, pa_fn, K, Q)
% ILA_TRAIN  Identify memory-polynomial predistorter via Indirect Learning.
%
% Implements the single-pass ILA (paper Sec. III, Fig. 1):
%   • Training block A has y(n)/G as input, x(n) as desired output.
%   • Predistorter is an exact copy of block A applied to x(n).
%
% Steps:
%   1. y  = PA(x_tr)                      [PA response to training input]
%   2. G  = sqrt( E[|y|²] / E[|x|²] )    [gain estimate, below Fig. 1]
%   3. U  = mp_basis(y/G, K, Q)           [postdistorter basis — eq. (5)]
%   4. â  = (UᴴU + λI)⁻¹ Uᴴ x_tr        [least-squares — eq. (7)]

    n_c = numel(1:2:K) * (Q + 1);

    % Step 1 — PA response to the training input
    y = pa_fn(x_tr);

    % Step 2 — Gain estimate [below Fig. 1]
    G = sqrt(mean(abs(y).^2) / mean(abs(x_tr).^2));

    % Step 3 — Postdistorter basis from scaled PA output — eq. (5)
    %   u_{kq}(n) = [y(n-q)/G] · |y(n-q)/G|^{k-1}
    U = mp_basis(y / G, K, Q);

    % Step 4 — Least-squares solution — eq. (7)
    %   â = (UᴴU)⁻¹ Uᴴ x_tr
    %   Tikhonov regularisation (λ·I) for numerical stability.
    lambda = 1e-4 * mean(diag(real(U' * U)));
    a = (U' * U + lambda * eye(n_c)) \ (U' * x_tr);
end

% ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

function y = pa_wh(u, H_b, H_a, G_b, G_a, b1, b3, b5)
% PA_WH  Wiener-Hammerstein power amplifier model (paper Fig. 2).
%
%   u(n) → H(z) → v(n) → F(v) → w(n) → G(z) → y(n)
%
%   H(z), G(z) : LTI filters,        eq. (8)
%   F(v)       : memoryless NL,       eq. (9)
%   b1,b3,b5   : NL coefficients,     eq. (10)

    v = filter(H_b, H_a, u);                          % H(z)  — eq. (8)
    w = b1*v + b3*v.*abs(v).^2 + b5*v.*abs(v).^4;    % F(v)  — eq. (9)
    y = filter(G_b, G_a, w);                          % G(z)  — eq. (8)
end

% ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

function Phi = mp_basis(u, K, Q)
% MP_BASIS  Build the memory-polynomial basis matrix.
%
%   Implements the basis of eq. (4) / eq. (5):
%     column (k,q): u(n-q) · |u(n-q)|^{k-1}
%
%   k ∈ {1, 3, 5, …, K}  (odd orders only — bandpass PA convention)
%   q ∈ {0, 1, …, Q}     (integer sample delays)
%
%   Returns Phi : N_u × [numel(odd_k)·(Q+1)]  complex matrix.

    N_u   = length(u);
    odd_k = 1:2:K;
    Phi   = zeros(N_u, numel(odd_k)*(Q+1), 'like', 1j);
    col   = 1;
    for k = odd_k
        for q = 0:Q
            ud         = [zeros(q, 1, 'like', u); u(1:N_u-q)];  % delay by q
            Phi(:,col) = ud .* abs(ud).^(k-1);                   % eq. (4)/(5)
            col        = col + 1;
        end
    end
end
