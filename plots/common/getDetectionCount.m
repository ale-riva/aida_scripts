
%  Helper function to replace zeros with NaN

function count = getDetectionCount(d, x_rel)
    if isfield(d, 'count')
        count = d.count;
    else
        count = sum(isfinite(x_rel), 2);
    end
end