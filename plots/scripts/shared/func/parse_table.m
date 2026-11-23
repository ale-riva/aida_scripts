function [labels] = parse_table(T, field_names)
    % Copy the requested table columns into same-named struct fields.
    labels = struct();
    for i = 1:numel(field_names)
        field_name = field_names{i};
        labels.(field_name) = T.(field_name);
    end
end
