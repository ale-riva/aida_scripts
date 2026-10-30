function out = filter_by_class(det_struct, class_ids)
%FILTER_BY_CLASS Filter detections by class and compact each message.
%
% Rows represent messages, columns represent detections.
% The output preserves the number of messages and reduces the number
% of columns to the maximum number of simultaneous filtered detections.

% Identify detections to keep
to_keep = ismember(det_struct.class, class_ids) & ~isnan(det_struct.x_cog);

% Compute detection counts
n_msgs = size(det_struct.class, 1);
out.count = sum(to_keep, 2);
out.max_det = max([0; out.count]);

% Compute new column indices for each detection
new_cols = cumsum(to_keep, 2);

% Compute linear indices in the compacted arrays
[rows, ~] = find(to_keep);

if out.max_det > 0
    dst_idx = sub2ind([n_msgs, out.max_det], rows, new_cols(to_keep));
else
    dst_idx = [];
end

% Fields to filter
fields = {'x_cog', 'y_cog', 'z_cog', 'orientation', ...
          'height', 'width','length', 'score', 'class', 'stamp'};

% Filter and compact each field
for i = 1:numel(fields)
    field = fields{i};
    src = det_struct.(field);

    dst = nan(n_msgs, out.max_det);

    if strcmp(field, 'stamp') && ...
            isvector(src) && numel(src) == n_msgs && ...
            ~isequal(size(src), size(to_keep))
        % One timestamp per message
        dst(dst_idx) = src(rows);
    else
        % One value per detection
        dst(dst_idx) = src(to_keep);
    end

    out.(field) = dst;
end

end
