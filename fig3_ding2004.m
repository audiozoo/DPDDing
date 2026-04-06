% fig3_ding2004.m
%
% Reproduces Figure 3 from:
%   L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim, C. R. Giardina,
%   "A robust digital baseband predistorter constructed using memory polynomials,"
%   IEEE Trans. Commun., vol. 52, no. 1, pp. 159–165, Jan. 2004.
%
% Figure 3: Simulated output PSD for a 3-carrier WCDMA uplink signal comparing:
%   (1) Linear (desired) output
%   (2) PA output without predistortion
%   (3) PA output with memoryless DPD        (K=5, Q=0)
%   (4) PA output with memory polynomial DPD (K=5, Q=2)
%
% PA model: Hammerstein — 5th-order complex polynomial nonlinearity followed
%           by a short FIR memory filter.
% DPD training: Indirect Learning Architecture (ILA), least-squares.

clear; clc; close all;
rng(42);

%% ── Parameters ───────────────────────────────────────────────────────────
fs          = 61.44e6;   % Sample rate (16x WCDMA chip rate 3.84 Mcps)
chip_rate   = 3.84e6;    % WCDMA chip rate
carrier_sep = 5e6;       % 5 MHz carrier spacing
N           = 2^16;      % Number of samples
K_ml        = 5;         % Polynomial order, memoryless DPD  (odd: 1,3,5)
K_mp        = 5;         % Polynomial order, memory poly DPD
Q_mp        = 2;         % Memory depth,    memory poly DPD
fir_len     = 127;       % Carrier-shaping FIR length

%% ── PA model coefficients ────────────────────────────────────────────────
% 5th-order complex polynomial + 3-tap FIR memory
c1   =  1.00 + 0.00j;
c3   = -0.20 - 0.15j;
c5   =  0.05 + 0.04j;
h_pa = [1.0,  0.15-0.10j,  0.05+0.05j];

%% ── Generate 3-carrier WCDMA-like baseband signal ────────────────────────
% Each carrier: LPF-filtered complex Gaussian noise modulated to ±5 MHz, 0 Hz.
bw_norm = (chip_rate * 1.25) / (fs/2);  % per-carrier BW normalised to Nyquist
h_lp    = fir1(fir_len-1, bw_norm/2);  % single-carrier lowpass shaping filter
n_vec   = (0:N-1).';
x = zeros(N,1);
for fc = [-carrier_sep, 0, carrier_sep]
    raw   = (randn(N,1) + 1j*randn(N,1)) / sqrt(2);
    lp    = filter(h_lp, 1, raw);
    x     = x + lp .* exp(1j*2*pi*(fc/fs)*n_vec);
end
x = x / (0.95 * max(abs(x)));   % peak normalise

%% ── PA output without DPD ────────────────────────────────────────────────
y_no_dpd = pa_model(x, c1, c3, c5, h_pa);

%% ── ILA: train postdistorter, use copy as predistorter ───────────────────
% Estimate PA linear gain for target scaling
gain  = (x' * y_no_dpd) / (x' * x);
x_tgt = x * gain;

% ── Memoryless DPD (K=5, Q=0) ──────────────────────────────────────────
Phi_ml   = mp_basis(y_no_dpd, K_ml, 0);
a_ml     = (Phi_ml' * Phi_ml) \ (Phi_ml' * x_tgt);
y_ml     = pa_model(mp_basis(x, K_ml, 0) * a_ml, c1, c3, c5, h_pa);

% ── Memory polynomial DPD (K=5, Q=2) ───────────────────────────────────
Phi_mp   = mp_basis(y_no_dpd, K_mp, Q_mp);
a_mp     = (Phi_mp' * Phi_mp) \ (Phi_mp' * x_tgt);
y_mp     = pa_model(mp_basis(x, K_mp, Q_mp) * a_mp, c1, c3, c5, h_pa);

%% ── PSD via Welch ─────────────────────────────────────────────────────────
nfft     = 4096;
win      = hann(nfft);
noverlap = nfft/2;

[P_lin,   f_hz] = pwelch(x,        win, noverlap, nfft, fs, 'twosided');
[P_noDPD, ~   ] = pwelch(y_no_dpd, win, noverlap, nfft, fs, 'twosided');
[P_ml,    ~   ] = pwelch(y_ml,     win, noverlap, nfft, fs, 'twosided');
[P_mp,    ~   ] = pwelch(y_mp,     win, noverlap, nfft, fs, 'twosided');

% Shift to centred axis: -fs/2 … +fs/2
f_shift = f_hz - fs*(f_hz >= fs/2);
[f_MHz, sort_idx] = sort(f_shift / 1e6);

P = @(raw) raw(sort_idx);            % reorder any PSD vector
ref_lin = max(P(P_lin));             % reference power for 0 dB top
dBrel   = @(raw) 10*log10(P(raw) / ref_lin);

%% ── Plot ─────────────────────────────────────────────────────────────────
figure('Color','w', 'Position',[100 100 720 500]);

plot(f_MHz, dBrel(P_lin),   'k-',  'LineWidth', 1.3); hold on;
plot(f_MHz, dBrel(P_noDPD), 'r--', 'LineWidth', 1.3);
plot(f_MHz, dBrel(P_ml),    'b-.', 'LineWidth', 1.3);
plot(f_MHz, dBrel(P_mp),    'g-',  'LineWidth', 1.6);

xlim([-25 25]);  ylim([-80 5]);
grid on;  box on;
xlabel('Frequency (MHz)',       'FontSize', 12);
ylabel('Normalised PSD (dB)',   'FontSize', 12);
title({'Simulated PA Output PSD — 3-Carrier WCDMA Signal'; ...
       'Ding et al. 2004, Fig. 3'},  'FontSize', 12);
legend({'Linear (desired)', ...
        'No predistortion', ...
        'Memoryless DPD (K=5, Q=0)', ...
        'Memory poly. DPD (K=5, Q=2)'}, ...
       'Location','south', 'FontSize', 10);

for fc = [-10 -5 0 5 10]
    xline(fc, ':', 'Color', [0.7 0.7 0.7]);   % carrier centre markers
end

%% ── ACPR report ──────────────────────────────────────────────────────────
% Average adjacent-channel power relative to total in-band power
acpr = @(P_s, P_d) calc_acpr(f_MHz, P(P_s), P(P_d));
fprintf('\n--- ACPR (adjacent channels ±10–20 MHz) ---\n');
fprintf('  No DPD           : %+.1f dBc\n', acpr(P_lin, P_noDPD));
fprintf('  Memoryless DPD   : %+.1f dBc\n', acpr(P_lin, P_ml));
fprintf('  Memory poly. DPD : %+.1f dBc\n', acpr(P_lin, P_mp));

% =========================================================================
%  LOCAL FUNCTIONS  (must appear after all script body code)
% =========================================================================

function y = pa_model(u, c1, c3, c5, h_pa)
% Hammerstein PA: static 5th-order polynomial then FIR memory filter.
    nl = c1*u + c3*u.*abs(u).^2 + c5*u.*abs(u).^4;
    y  = filter(h_pa, 1, nl);
end

function Phi = mp_basis(u, K, Q)
% Build the memory-polynomial basis matrix for signal u.
% Columns: odd orders k=1,3,...,K crossed with delays q=0,...,Q.
    N_u    = length(u);
    odd_k  = 1:2:K;
    Phi    = zeros(N_u, numel(odd_k)*(Q+1), 'like', 1j);
    col = 1;
    for k = odd_k
        for q = 0:Q
            ud          = [zeros(q,1,'like',u); u(1:N_u-q)];
            Phi(:, col) = ud .* abs(ud).^(k-1);
            col         = col + 1;
        end
    end
end

function acpr_dBc = calc_acpr(f_MHz, P_sig, P_dist)
% ACPR: mean distorted power in adjacent channels vs mean signal power in-band.
% In-band: |f| < 9 MHz (covers all 3 WCDMA carriers at ±5 MHz).
% Adjacent: 10 MHz < |f| < 20 MHz.
    inband = abs(f_MHz) < 9;
    adj    = abs(f_MHz) >= 10 & abs(f_MHz) <= 20;
    acpr_dBc = 10*log10(mean(P_dist(adj)) / mean(P_sig(inband)));
end
