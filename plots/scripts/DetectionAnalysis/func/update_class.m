function [out_det] = update_class(input_det, annot)
    % UPDATE_CLASS  Update detection classes using manual annotations.
    %
    %   Column j in annot.class corresponds to column j in input_det.class.
    %   Match rows by timestamp; neither struct needs an id field.
    %   Annotation matrices may have fewer columns than detection matrices.
    %   Only valid (non-NaN) annotations overwrite the original classes.
    %   Unmatched timestamps and unannotated columns retain their values.
    %   Overwriting a NaN detection class raises an error.

    out_det = input_det;

    n_annot_cols = size(annot.class, 2);
    if n_annot_cols > size(input_det.class, 2) || ...
            numel(annot.stamp) ~= size(annot.class, 1)
        error('update_class:InvalidDimensions', ...
            ['Annotation timestamps must have one entry per row, and ' ...
             'annotations cannot have more columns than detections.']);
    end
    if numel(unique(annot.stamp)) ~= numel(annot.stamp)
        error('update_class:DuplicateTimestamps', ...
            'Annotation timestamps must be unique.');
    end

    % Expand timestamps if necessary
    if isvector(out_det.stamp) && ...
            numel(out_det.stamp) == size(out_det.class, 1)
        stamps = repmat(out_det.stamp(:), 1, size(out_det.class, 2));
    else
        stamps = out_det.stamp;
    end
    if ~isequal(size(stamps), size(out_det.class))
        error('update_class:InvalidDimensions', ...
            'Detection timestamps must be per message or per detection.');
    end

    % Match timestamps only within the corresponding annotation columns.
    stamps = stamps(:, 1:n_annot_cols);
    [matched_stamp, annot_rows] = ismember(stamps, annot.stamp(:));
    matched = matched_stamp & isfinite(stamps);

    % Get matched row and column indices
    [rows, cols] = find(matched);

    % The destination and source use the same column number.
    dst_idx = sub2ind(size(out_det.class), rows, cols);
    src_idx = sub2ind(size(annot.class), annot_rows(matched), cols);

    new_classes = annot.class(src_idx);

    % Update only non-NaN annotations
    to_update = ~isnan(new_classes);

    % Check that no NaN detection class is overwritten
    invalid = to_update & isnan(out_det.class(dst_idx));

    if any(invalid)
        idx = find(invalid, 1);
        error('update_class:InvalidOverwrite', ...
            'Cannot overwrite NaN detection at row %d, column %d.', ...
            rows(idx), cols(idx));
    end

    % Apply annotations
    out_det.class(dst_idx(to_update)) = new_classes(to_update);

end
