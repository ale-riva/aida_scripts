function out = replaceZeroWithNaN(x)
    out = x;
    out(out == 0) = NaN;
end