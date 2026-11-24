function out = filter_cam_det_by_class(cam_det, class_ids)
    % FILTER_CAM_DET_BY_CLASS  Filter camera detections by class without compacting.
    %
    % Rows represent messages, columns represent detections.
    % The output preserves the original matrix dimensions.
    % Detections not belonging to the specified classes are set to NaN.

    % Copy original struct
    out = cam_det;

    camera_names = fieldnames(cam_det);

    for i = 1:numel(camera_names)
        camera_name = camera_names{i};
        det = cam_det.(camera_name);

        % Identify detections to keep
        to_keep = ismember(det.class, class_ids);

        % Update detection counts
        out.(camera_name).count = sum(to_keep, 2);
        out.(camera_name).max_det = max([0; out.(camera_name).count]);

        % Fields to filter
        fields = {'id', 'class', 'score', 'sensor_type'};

        % Filter each field without changing dimensions
        for j = 1:numel(fields)
            field = fields{j};

            if ~isfield(det, field)
                continue;
            end

            src = det.(field);
            src(~to_keep) = NaN;

            out.(camera_name).(field) = src;
        end

        % Filter class names
        if isfield(det, 'class_name')
            out.(camera_name).class_name(~to_keep) = "";
        end

        % Filter bounding box fields
        if isfield(det, 'bbox')
            fields = fieldnames(det.bbox);

            for j = 1:numel(fields)
                field = fields{j};

                src = det.bbox.(field);
                src(~to_keep) = NaN;

                out.(camera_name).bbox.(field) = src;
            end
        end
    end
end