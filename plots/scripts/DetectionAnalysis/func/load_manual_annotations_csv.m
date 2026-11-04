function [labels] = load_manual_annotations_csv(T)
    % CSV class columns must be in the same order as lid_det.class columns.
    % Detection identity is given by column position, without an id field.
    labels = struct();
    labels.stamp = double(T{:, 1});        % Nx1 double (s)
    labels.class = T{:, 2:end};    % NxM double (NaN for missing detections)
end
