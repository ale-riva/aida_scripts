% Count detection observations for VIS_CLASSES, not unique objects.
class_names = fieldnames(DET_LABELS);

for i = 1:numel(sensors)
    s = sensors{i}.s;
    valid = isfinite(s.stamp) & isfinite(s.x_cog);

    fprintf('\nDetection totals - %s\n',sensors{i}.name);
    fprintf('%-16s %10s\n','Class','Detections');

    total_count = 0;
    for j = 1:numel(class_names)
        class_name = class_names{j};
        if ~ismember(DET_LABELS.(class_name),VIS_CLASSES)
            continue;
        end

        class_count = nnz(valid & s.class == DET_LABELS.(class_name));
        fprintf('%-16s %10d\n',class_name,class_count);
        total_count = total_count + class_count;
    end

    fprintf('%-16s %10d\n','Total (selected)',total_count);
end
