

% Draw Full path
figure("Name","CONFIDENCE",'NumberTitle','off')
tiledlayout(3,1,"Padding","compact");


% pdf
nexttile([2,1]); hold on;
normalization = 'pdf';
class_names = fieldnames(DET_LABELS);
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = isfinite(s.stamp) & isfinite(s.score);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        max_pt = max(s.score(class_idx));
        
        histogram( ...
            s.score(class_idx), ...
            'BinWidth', 0.05, ...
            'Normalization', normalization, ...
            'DisplayStyle', 'stairs', ...
            'LineWidth', 1.5, ...
            'EdgeColor', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));
    end
end
xlabel("confidence [-]")
ylabel([normalization  "[-]"])

% count
nexttile;hold on;
normalization = "count";
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = isfinite(s.stamp) & isfinite(s.score);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        max_pt = max(s.score(class_idx));

        histogram( ...
            s.score(class_idx), ...
            'BinWidth', 0.05, ...
            'Normalization', normalization, ...
            'DisplayStyle', 'stairs', ...
            'LineWidth', 1.5, ...
            'EdgeColor', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
            sensors{i}.name, class_name));
    end
end
xlabel("confidence [-]")
ylabel([normalization  "[-]"])
legend show
