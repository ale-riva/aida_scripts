function out = unpackField(d, field_name, count, n_msg, n_det)
% UNPACKFIELD  Convert variable-length cell arrays to NaN-padded matrices.
%
%   Each cell represents one message and contains the values of
%   the detections associated with that message.
%
%   Output:
%       out: [n_msg x n_det] double matrix
%
%   Missing fields and unused detection slots are filled with NaN.

out = NaN(n_msg, n_det);

if ~isfield(d, field_name)
    return;
end

data = d.(field_name);

for i = 1:n_msg
    values = data{i};

    if isempty(values)
        continue;
    end

    values = double(values(:));

    n = min(numel(values), count(i));

    if n > 0
        out(i, 1:n) = values(1:n);
    end
end

end
