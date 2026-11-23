function [labels] = parse_table(T, field_name)
    % CSV class columns must be in the same order as lid_det.class columns.
    % Detection identity is given by column position, without an id field.
    labels = struct();
    labels.stamp = double(T{:, 1});        % Nx1 double (s)
    labels.(field_name) = T{:, 2:end};    % NxM double (NaN for missing detections)
end
