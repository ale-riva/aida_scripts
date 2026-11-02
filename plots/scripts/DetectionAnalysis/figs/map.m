

% Draw Full path
figure("Name","MAP",'NumberTitle','off')
hold on; axis equal;
xlabel("X [m]")
ylabel("Y [m]")
plot(ego.x,ego.y,'LineWidth',0.3,'Color',[0.5 0.5 0.5],'DisplayName','Ego')
t1_start = 0;
% idx = 100; % Detection message to display.

 % --- sensor detections ---
class_names = fieldnames(DET_LABELS);
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = ...
        isfinite(s.stamp) & ...
        isfinite(s.x_map) & ...
        isfinite(s.y_map);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        plot( ...
            s.x_map(class_idx), ...
            s.y_map(class_idx), ...
            '.', ...
            'MarkerSize', 20, ...
            'Color', col.(class_name), ...
            'DisplayName', sprintf('%s - %s',sensors{i}.name,class_name));
    end
end
legend show
