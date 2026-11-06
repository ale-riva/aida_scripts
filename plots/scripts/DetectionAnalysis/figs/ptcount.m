

% Draw Full path
figure("Name","PT_COUNT",'NumberTitle','off')
tiledlayout(3,1,"Padding","compact");


% pt_count histogram 
nexttile([2,1]); hold on;
normalization = 'pdf';
class_names = fieldnames(DET_LABELS);
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = isfinite(s.stamp) & isfinite(s.pt_count);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        max_pt = max(s.pt_count(class_idx));

        bin_edges = [ ...
            0:5:400, ...
            420:20:max(420, ceil(max_pt/20)*20)];
        
        histogram( ...
            s.pt_count(class_idx), ...
            'BinWidth', 5, ...
            'Normalization', normalization, ...
            'DisplayStyle', 'stairs', ...
            'LineWidth', 1.5, ...
            'EdgeColor', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));
    end
end
xlabel("pt count [-]")
ylabel([normalization  "[-]"])

% confidence histogram
nexttile; hold on;
normalization = 'count';
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = isfinite(s.stamp) & isfinite(s.pt_count);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        max_pt = max(s.score(class_idx));

        histogram( ...
            s.pt_count(class_idx), ...
            'BinWidth', 5, ...
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
