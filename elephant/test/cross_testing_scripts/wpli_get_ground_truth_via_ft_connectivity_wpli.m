%% use FieldTrips' ft_connectivity_wpli() to calculate WPLI ground-truth

function wpli_get_ground_truth_via_ft_connectivity_wpli(data_type)
    addpath '/path/to/FieldTrip_toolbox'

    if strcmpi(data_type, "REAL")
        % cutted REAL DATA from multielectrode_grasp GIN-repository 
        cs_dataset = load('i140703-001_cross_spectrum_of_channel_1_and_2_of_slice_TS_ON_to_GO_ON_corect_trials.mat');
        cs = cs_dataset.cross_spectrum_matrix;
        disp("REAL data given");
    elseif strcmpi(data_type, "ARTIFICIAL")
        % ARTIFICIAL DATA more complex
        cs_dataset =load('cross_spectrum_of_artificial_LFPs_1_and_2.mat');
        cs = cs_dataset.cross_spectrum_matrix;
        disp("ARTIFICIAL data given");
    else
        disp("SOMETHING data given");
        throw(ValueError("'REAL' and 'ARTIFICIAL' are accepted types"));
    end

    % % if cross-spectrum is directly used from loaded file, this final reshape is needed
    siz_2 = size(cs)
    trials = siz_2(1)
    freq_points = siz_2(2)
    cs = reshape(cs, [trials, 1, freq_points]);


    [wpli_1, v, n] = ft_connectivity_wpli(cs, 'feedback', 'text', 'dojack', 0, 'debias', 0);

    if strcmpi(data_type, "REAL")
        wpli_1 = abs(wpli_1); % take abs() for real data
        writematrix(wpli_1, 'ground_truth_WPLI_from_ft_connectivity_wpli_with_real_LFPs_R2G.csv'); 
    else
        writematrix(wpli_1, 'ground_truth_WPLI_from_ft_connectivity_wpli_with_artificial_LFPs.csv');
    end
    
    exit;
end
