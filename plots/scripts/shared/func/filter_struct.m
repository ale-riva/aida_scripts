function [out, keep] = filter_struct(s, field_name, condition)
% FILTER_STRUCT  Filter detections using a field and a numeric comparison.
%
%   out = filter_struct(s, 'pt_count', '>=50');
%   out = filter_struct(s, 'score', '<=0.7');
%   [out, keep] = filter_struct(s, 'height', '>1.5');
%
%   The selected field must be a real numeric matrix with one row per frame
%   and one column per detection. Operators: >, >=, <, <=, == and ~=.
%   Thresholds support signs, decimals and scientific notation. NaN and Inf
%   values in the selected field are always excluded.
%
%   Excluded detections become NaN in numeric/logical fields with matching
%   first two dimensions, including arrays in nested scalar structs (bbox).
%   Integer/logical detection fields are converted to double when masked.
%   Rows and columns are preserved; keep is the logical detection mask.
%   Per-frame column-vector stamps, frame identifiers, scalar NaN placeholders
%   and metadata are preserved. Per-detection stamp matrices are masked.
%   Existing count and max_det fields are updated to reflect the kept entries.
%   The condition is parsed as a comparison, never evaluated as MATLAB code.

if ~isstruct(s) || ~isscalar(s)
    error('filter_struct:InvalidStruct', 'Input must be a scalar struct.');
end
if ~((ischar(field_name) && isrow(field_name)) || ...
        (isstring(field_name) && isscalar(field_name) && ~ismissing(field_name)))
    error('filter_struct:InvalidFieldName', ...
        'Field name must be a character row vector or a scalar string.');
end
field_name = char(field_name);
if ~isfield(s,field_name)
    error('filter_struct:MissingField', 'Input has no field named %s.',field_name);
end
values = s.(field_name);
if ~isnumeric(values) || ~isreal(values) || ~ismatrix(values)
    error('filter_struct:InvalidField', ...
        'The selected field must be a real numeric matrix.');
end
% Compare integer fields against fractional thresholds without rounding them.
values = double(values);
if ~((ischar(condition) && isrow(condition)) || ...
        (isstring(condition) && isscalar(condition) && ~ismissing(condition)))
    error('filter_struct:InvalidCondition', ...
        'Condition must be a character row vector or a scalar string.');
end

pattern = '^\s*(>=|<=|==|~=|>|<)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$';
tokens = regexp(char(condition),pattern,'tokens','once');
if isempty(tokens)
    error('filter_struct:InvalidCondition', ...
        'Use a numeric comparison such as >=50 or <=0.7.');
end
threshold = str2double(tokens{2});
if ~isfinite(threshold)
    error('filter_struct:InvalidCondition', 'Threshold must be finite.');
end

switch tokens{1}
    case '>'
        keep = values > threshold;
    case '>='
        keep = values >= threshold;
    case '<'
        keep = values < threshold;
    case '<='
        keep = values <= threshold;
    case '=='
        keep = values == threshold;
    case '~='
        keep = values ~= threshold;
end
keep = keep & isfinite(values);

out = mask_fields(s,keep);
if isfield(s,'count')
    out.count = sum(keep,2);
end
if isfield(s,'max_det')
    out.max_det = max([0; sum(keep,2)]);
end

end

function out = mask_fields(s,keep)
% Apply a detection mask without changing frame metadata or array shapes.
out = s;
fields = fieldnames(s);
metadata = {'count','max_det','max_tracks','frame_id','frame_id_rel','frame_id_map'};
for i = 1:numel(fields)
    field = fields{i};
    value = s.(field);
    if ismember(field,metadata)
        continue;
    end
    if isstruct(value) && isscalar(value)
        out.(field) = mask_fields(value,keep);
        continue;
    end
    if ~(isnumeric(value) || islogical(value)) || ...
            size(value,1) ~= size(keep,1) || size(value,2) ~= size(keep,2)
        continue;
    end
    if (strcmp(field,'stamp') && iscolumn(value)) || ...
            (isscalar(value) && isnan(value)) || ~any(~keep(:))
        continue;
    end
    if ~isfloat(value)
        value = double(value);
    end
    repetitions = size(value);
    repetitions(1:2) = 1;
    value(repmat(~keep,repetitions)) = NaN;
    out.(field) = value;
end
end
