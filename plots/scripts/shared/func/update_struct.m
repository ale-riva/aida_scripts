function [s1] = update_struct(s1, s2, keep_only_matched)
% UPDATE_STRUCT  Update matching struct fields by timestamp and column.
%
%   s2 must contain stamp and at least one other field, all present in s1.
%   Fields have one row per timestamp; s2 cannot have more rows or columns.
%   Stamps are column vectors. Per-detection stamps in s1 are also supported.
%   Scalar NaN fields are expanded using the width of s1.x_map, per-detection
%   stamps or the widest matching s1 field. Otherwise, use the widest s2 field.
%   Only non-NaN values in s2 overwrite s1. Unmatched values are preserved.
%   keep_only_matched defaults to false. When true, retain only rows with
%   at least one finite timestamp matching s2, even if its values are NaN.
%   Filter all row-aligned numeric/logical fields, preserving column positions,
%   frame identifiers, max_det and unexpanded scalar NaN placeholders.

if nargin < 3
    keep_only_matched = false;
end
validateattributes(keep_only_matched, {'logical','numeric'}, ...
    {'scalar','binary'}, mfilename, 'keep_only_matched', 3);


%  Check matching fields and timestamps

if ~isstruct(s1) || ~isscalar(s1) || ~isstruct(s2) || ~isscalar(s2)
    error('update_class:InvalidStructs', ...
        'Inputs must be scalar structs.');
end

fields = fieldnames(s2);
if ~isfield(s2,'stamp') || numel(fields) < 2 || ~all(isfield(s1,fields))
    error('update_class:MissingFields', ...
        's2 must contain stamp and another field, all present in s1.');
end
fields(strcmp(fields,'stamp')) = [];

if ~isnumeric(s1.stamp) || ~ismatrix(s1.stamp) || ...
        ~isnumeric(s2.stamp) || ~ismatrix(s2.stamp) || size(s2.stamp,2) ~= 1
    error('update_class:InvalidStamps', ...
        'Stamps must be numeric; s2.stamp must be a column vector.');
end

n1 = size(s1.stamp,1);
n2 = size(s2.stamp,1);
if n1 < n2
    error('update_class:InvalidDimensions', ...
        's2 cannot have more timestamp rows than s1.');
end
if numel(unique(s2.stamp)) ~= n2
    error('update_class:DuplicateTimestamps', ...
        's2 timestamps must be unique.');
end


%  Check dimensions and determine the width for scalar NaN fields

n_cols1 = [];
n_cols2 = 0;
if size(s1.stamp,2) > 1
    n_cols1 = size(s1.stamp,2);
end
if isfield(s1,'x_map') && isnumeric(s1.x_map) && ismatrix(s1.x_map) && ...
        ~(isscalar(s1.x_map) && isnan(s1.x_map))
    if size(s1.x_map,1) ~= n1
        error('update_class:InvalidDimensions', ...
            's1.x_map must have one row per timestamp.');
    end
    n_cols1(end+1) = size(s1.x_map,2);
end

for i = 1:numel(fields)
    field = fields{i};
    dst = s1.(field);
    src = s2.(field);
    if ~isnumeric(dst) || ~ismatrix(dst) || ~isnumeric(src) || ~ismatrix(src)
        error('update_class:InvalidFields', ...
            'Matching field %s must contain numeric matrices.',field);
    end
    if size(src,1) ~= n2
        error('update_class:InvalidDimensions', ...
            's2.%s must have one row per timestamp.',field);
    end
    if ~(isscalar(dst) && isnan(dst))
        if size(dst,1) ~= n1 || size(dst,2) < size(src,2)
            error('update_class:InvalidDimensions', ...
                's1.%s must have N1 rows and at least as many columns as s2.%s.',field,field);
        end
        n_cols1(end+1) = size(dst,2);
    end
    if size(s1.stamp,2) > 1 && size(s1.stamp,2) < size(src,2)
        error('update_class:InvalidDimensions', ...
            's1.stamp must cover all columns of s2.%s.',field);
    end
    n_cols2 = max(n_cols2,size(src,2));
end

if isempty(n_cols1)
    n_cols1 = n_cols2;
else
    n_cols1 = max(n_cols1);
end


%  Match timestamps and update corresponding columns

[matched, rows2] = ismember(s1.stamp,s2.stamp);
matched = matched & isfinite(s1.stamp);

for i = 1:numel(fields)
    field = fields{i};
    dst = s1.(field);
    src = s2.(field);
    if isscalar(dst) && isnan(dst)
        if n_cols1 < size(src,2)
            error('update_class:InvalidDimensions', ...
                's2.%s has more columns than the matching s1 fields.',field);
        end
        dst = nan(n1,n_cols1);
    end

    n_cols = size(src,2);
    if size(s1.stamp,2) == 1
        field_matched = repmat(matched,1,n_cols);
        field_rows2 = repmat(rows2,1,n_cols);
    else
        field_matched = matched(:,1:n_cols);
        field_rows2 = rows2(:,1:n_cols);
    end
    [rows, cols] = find(field_matched);
    dst_idx = sub2ind(size(dst),rows,cols);
    src_idx = sub2ind(size(src),field_rows2(field_matched),cols);
    values = src(src_idx);
    to_update = ~isnan(values);
    dst(dst_idx(to_update)) = values(to_update);
    s1.(field) = dst;
end

%  Remove unmatched rows consistently across all row-aligned fields

if keep_only_matched
    keep_rows = any(matched,2);
    all_fields = fieldnames(s1);
    metadata = {'max_det','frame_id','frame_id_rel','frame_id_map'};
    for i = 1:numel(all_fields)
        field = all_fields{i};
        value = s1.(field);
        if ismember(field,metadata) || ...
                ~(isnumeric(value) || islogical(value)) || size(value,1) ~= n1
            continue;
        end
        if isscalar(value) && isnan(value) && ...
                ~strcmp(field,'stamp') && ~ismember(field,fields)
            continue;
        end
        indices = repmat({':'},1,ndims(value));
        indices{1} = keep_rows;
        s1.(field) = value(indices{:});
    end
end

end % FUNCTION END
