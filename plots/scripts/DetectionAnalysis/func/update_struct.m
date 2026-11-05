function [s1] = update_struct(s1, s2)
% UPDATE_CLASS  Update matching struct fields by timestamp and column.
%
%   s2 must contain stamp and at least one other field, all present in s1.
%   Fields have one row per timestamp; s2 cannot have more rows or columns.
%   Stamps are column vectors. Per-detection stamps in s1 are also supported.
%   Scalar NaN fields are expanded using the widest matching s1 field,
%   or the widest s2 field when all matching s1 fields are scalar NaN.
%   Only non-NaN values in s2 overwrite s1. Unmatched values are preserved.


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

end % FUNCTION END
