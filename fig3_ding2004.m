% fig3_ding2004.m
%
% Reproduces Figure 3 from:
%   L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim, C. R. Giardina,
%   "A robust digital baseband predistorter constructed using memory polynomials,"
%   IEEE Trans. Commun., vol. 52, no. 1, pp. 159–165, Jan. 2004.
%
% Figure 3 caption (from paper):
%   "Effectiveness of predistortion in suppressing spectral regrowth when
%    the PA is modeled by a W-H system.
%    (a) Output without predistortion.
%    (b) Output with memoryless predistortion.
%    (c) Output with memory polynomial predistortion (Q=2, K=5).
%    (d) Original input.  (c) and (d) almost coincide."
%
% PA MODEL — Wiener-Hammerstein (eq. 8, 9, 10):
%   x(n) → H(z) → v(n) → F(v) → w(n) → G(z) → y(n)
%
%   H(z) = (1 + 0.5 z^{-2}) / (1 - 0.2 z^{-1})          [eq. 8]
%   G(z) = (1 - 0.1 z^{-2}) / (1 - 0.4 z^{-1})          [eq. 8]
%
%   F(v): w(n) = b1*v + b3*v|v|^2 + b5*v|v|^4            [eq. 9]
%   b1 =  1.0108 + 0.0858j                               [eq. 10]
%   b3 =  0.0879 - 0.1583j
%   b5 = -1.0992 - 0.8891j
%     (extracted from an actual Class AB PA)
%
% PREDISTORTER — memory polynomial via Indirect Learning Architecture (ILA):
%   z(n) = sum_{k=1,3,5} sum_{q=0}^{Q} a_{kq} * x(n-q) * |x(n-q)|^{k-1}  [eq. 4]
%   Training: â = (U^H U)^{-1} U^H z  [eq. 7]
%   with basis u_{kq}(n) = (y(n-q)/G) * |y(n-q)/G|^{k-1}   [eq. 5]
%
% SIGNAL: 3-carrier WCDMA baseband, 8000 training samples (as stated in paper).

clear; clc; close all;
rng(42);

%% ── Parameters ───────────────────────────────────────────────────────────
fs        = 20e6;     % Sample rate — places 3 WCDMA carriers at ±0.5 norm. freq.
N         = 2^16;     % Total samples (for smooth PSD estimate)
N_train   = 8000;     % Training samples, as specified in paper Section V
K         = 5;        % Polynomial order (odd terms: 1, 3, 5)
Q_memless = 0;        % Memory depth — memoryless predistorter
Q_memory  = 2;        % Memory depth — memory polynomial predistorter
chip_rate = 3.84e6;   % WCDMA chip rate

%% ── Exact W-H PA model (eqs 8–10) ───────────────────────────────────────
% H(z) = (1 + 0.5 z^{-2}) / (1 - 0.2 z^{-1})
H_b = [1, 0, 0.5];   H_a = [1, -0.2];

% G(z) = (1 - 0.1 z^{-2}) / (1 - 0.4 z^{-1})
G_b = [1, 0, -0.1];  G_a = [1, -0.4];

% Memoryless nonlinearity F(v) coefficients
b1 =  1.0108 + 0.0858j;
b3 =  0.0879 - 0.1583j;
b5 = -1.0992 - 0.8891j;

%% ── Generate 3-carrier WCDMA-like baseband signal ────────────────────────
% Carriers at -5 MHz, 0 Hz, +5 MHz (5 MHz WCDMA spacing).
% Each: LPF-shaped complex Gaussian noise (statistically equivalent to
% filtered QPSK at chip_rate = 3.84 Mcps).
bw_norm = chip_rate / (fs/2);        % Per-carrier BW normalised to Nyquist
h_lp    = fir1(127, bw_norm/2);      % Single-carrier shaping lowpass filter
n_vec   = (0:N-1).';
x = zeros(N, 1);
for fc_hz = [-5e6, 0, 5e6]
    raw = (randn(N,1) + 1j*randn(N,1)) / sqrt(2);
    x   = x + filter(h_lp, 1, raw) .* exp(1j*2*pi*(fc_hz/fs)*n_vec);
end
% Set RMS so PA operates in its nonlinear regime
x = x * 0.3 / rms(x);

%% ── PA output (no predistortion) ─────────────────────────────────────────
y_nodpd = pa_wh(x, H_b,H_a, G_b,G_a, b1,b3,b5);

%% ── ILA predistorter training on first N_train samples ───────────────────
x_tr = x(1:N_train);
y_tr = y_nodpd(1:N_train);

% Estimate PA linear gain G (LS estimate: min ||G*x - y||^2)
G_est = real(x_tr' * y_tr) / real(x_tr' * x_tr);

% ── (b) Memoryless predistorter: K=5, Q=0 ──────────────────────────────
% Basis from y/G per eq. (5), target = x per eq. (6)
U_ml = mp_basis(y_tr/G_est, K, Q_memless);
a_ml = (U_ml' * U_ml) \ (U_ml' * x_tr);
y_ml = pa_wh(mp_basis(x, K, Q_memless)*a_ml, H_b,H_a, G_b,G_a, b1,b3,b5);

% ── (c) Memory polynomial predistorter: K=5, Q=2 ───────────────────────
U_mp = mp_basis(y_tr/G_est, K, Q_memory);
a_mp = (U_mp' * U_mp) \ (U_mp' * x_tr);
y_mp = pa_wh(mp_basis(x, K, Q_memory)*a_mp, H_b,H_a, G_b,G_a, b1,b3,b5);

%% ── Power spectral density (Welch) ───────────────────────────────────────
nfft     = 4096;
win      = hann(nfft);
noverlap = nfft/2;

[P_x,  f_hz] = pwelch(x,       win, noverlap, nfft, fs, 'twosided');
[P_nd, ~    ] = pwelch(y_nodpd, win, noverlap, nfft, fs, 'twosided');
[P_ml, ~    ] = pwelch(y_ml,    win, noverlap, nfft, fs, 'twosided');
[P_mp, ~    ] = pwelch(y_mp,    win, noverlap, nfft, fs, 'twosided');

% Shift to centred axis and normalise to ±1 (paper x-axis convention)
f_c    = f_hz - fs*(f_hz >= fs/2);
f_norm = f_c / (fs/2);
[f_norm, sidx] = sort(f_norm);

dBrel = @(P) 10*log10(P(sidx) / max(P_x));   % normalised to input peak

%% ── Figure ───────────────────────────────────────────────────────────────
figure('Color','w', 'Position',[100 100 680 520]);

plot(f_norm, dBrel(P_nd), 'k-',  'LineWidth', 1.3); hold on;
plot(f_norm, dBrel(P_ml), 'b--', 'LineWidth', 1.3);
plot(f_norm, dBrel(P_mp), 'r-',  'LineWidth', 1.5);
plot(f_norm, dBrel(P_x),  'k:',  'LineWidth', 1.0);

xlim([-1 1]);
ylim([-100 10]);
grid on; box on;

xlabel('Normalized Frequency', 'FontSize', 12);
ylabel('PSD (dB)',              'FontSize', 12);
title({'Effectiveness of predistortion in suppressing spectral regrowth'; ...
       '(W–H PA model)  —  Ding et al. 2004, Fig. 3'}, 'FontSize', 11);

legend({'(a) No predistortion', ...
        '(b) Memoryless predistortion', ...
        '(c) Memory poly. predistortion (Q=2, K=5)', ...
        '(d) Original input'}, ...
       'Location', 'south', 'FontSize', 10);

% =========================================================================
%  LOCAL FUNCTIONS  (all local functions must appear after script body)
% =========================================================================

function y = pa_wh(u, H_b,H_a, G_b,G_a, b1,b3,b5)
% Wiener-Hammerstein PA model (Fig. 2 / eq. 8-10 of Ding 2004):
%   u → H(z) → v → F(v) → w → G(z) → y
    v = filter(H_b, H_a, u);
    w = b1*v + b3*v.*abs(v).^2 + b5*v.*abs(v).^4;
    y = filter(G_b, G_a, w);
end

function Phi = mp_basis(u, K, Q)
% Memory-polynomial basis matrix (eq. 4-5 of Ding 2004).
% Columns: odd orders k=1,3,...,K  ×  delays q=0,...,Q.
% Row n:   [u(n)|u(n)|^0, u(n-1)|u(n-1)|^0, ..., u(n)|u(n)|^4, ...]
    N_u   = length(u);
    odd_k = 1:2:K;
    Phi   = zeros(N_u, numel(odd_k)*(Q+1), 'like', 1j);
    col   = 1;
    for k = odd_k
        for q = 0:Q
            ud         = [zeros(q,1,'like',u); u(1:N_u-q)];
            Phi(:,col) = ud .* abs(ud).^(k-1);
            col        = col + 1;
        end
    end
end
