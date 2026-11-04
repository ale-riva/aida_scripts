function out = filter_det_by_class(det_struct, class_ids)
    % FILTER_DET_BY_CLASS  Filter detections by class without compacting.
    %
    % Rows represent messages, columns represent detections.
    % The output preserves the original matrix dimensions.
    % Detections not belonging to the specified classes are set to NaN.

    % Identify detections to keep
    to_keep = ismember(det_struct.class, class_ids);

    % Copy original struct
    out = det_struct;

    % Update detection counts
    out.count = sum(to_keep, 2);
    out.max_det = max([0; out.count]);

labbel    % Fields to filter
    fields = {'x_map', 'y_map', 'z_map', 'roll_map', 'pitch_map', 'orientation_map', ...
              'x_cog', 'y_cog', 'z_cog', 'orientation', ...
              'height', 'width', 'length', 'score', 'class', 'id'};

    % Filter each field without changing dimensions
    for i = 1:numel(fields)
        field = fields{i};

        if ~isfield(det_struct, field)
            continue;
        end

        src = det_struct.(field);

        % Set excluded detections to NaN
        src(~to_keep) = NaN;

        out.(field) = src;
    end

end
