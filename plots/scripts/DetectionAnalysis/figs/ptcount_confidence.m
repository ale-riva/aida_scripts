% Confidence distributions per VIS_CLASSES, using the current filtered sensors.
% One row per sensor and one column per class, with linked XY axes.
% A shared logarithmic color scale shows raw observation counts.
% Populated cells are fully opaque; empty cells are transparent.

class_ids = unique(VIS_CLASSES(:), 'stable');
label_names = fieldnames(DET_LABELS);
label_ids = cellfun(@(name) DET_LABELS.(name), label_names);
[known, label_idx] = ismember(class_ids, label_ids);
if any(~known)
    warning('ptcount_confidence:UnknownClass', ...
        'Ignoring class IDs absent from DET_LABELS: %s.', mat2str(class_ids(~known).'));
end
class_ids = class_ids(known);
class_names = label_names(label_idx(known));

ptcount_confidence_figure = figure("Name","PTCOUNT-CONFIDENCE",'NumberTitle','off');
ptcount_confidence_axes = gobjects(numel(sensors), numel(class_ids));
if isempty(sensors) || isempty(class_ids)
    ax_empty = axes('Parent', ptcount_confidence_figure);
    axis(ax_empty, 'off');
    text(ax_empty, 0.5, 0.5, 'No sensors or classes selected.', ...
        'HorizontalAlignment', 'center');
else

    % Valid sensor detections
    pairs = cell(numel(sensors), numel(class_ids));
    max_points = 1;
    for i = 1:numel(sensors)
        s = sensors{i}.s;
        valid = isfinite(s.stamp) & isfinite(s.score) & isfinite(s.pt_count) & ...
            s.pt_count >= 0 & s.score >= 0 & s.score <= 1;
        for j = 1:numel(class_ids)
            selected = valid & s.class == class_ids(j);
            points = double(s.pt_count(selected));
            scores = double(s.score(selected));
            pairs{i,j} = [points(:), scores(:)];
            max_points = max([max_points; points(:)]);
        end
    end

    % Shared bins
    point_edges = linspace(0, log10(1 + max_points), 33);
    confidence_edges = linspace(0, 1, 21);
    point_centers = (point_edges(1:end-1) + point_edges(2:end))/2;
    confidence_centers = (confidence_edges(1:end-1) + confidence_edges(2:end))/2;
    ticks = unique([0, 1, 10.^(1:floor(log10(max_points)))]);
    ticks = ticks(ticks <= max_points);
    layout = tiledlayout(ptcount_confidence_figure, numel(sensors), numel(class_ids), ...
        'Padding', 'compact', 'TileSpacing', 'compact');
    max_count = 1;

    % --- Sensor detections ---
    for i = 1:numel(sensors)
        for j = 1:numel(class_ids)
            ax_heatmap = nexttile(layout);
            ptcount_confidence_axes(i,j) = ax_heatmap;
            data = pairs{i,j};
            counts = histcounts2(log10(1 + data(:,1)), data(:,2), ...
                point_edges, confidence_edges);
            max_count = max([max_count; counts(:)]);

            class_color = col.det.(class_names{j});
            img = imagesc(ax_heatmap, point_centers, confidence_centers, counts.');
            colormap(ax_heatmap, turbo(256));
            img.AlphaData = double(counts.' > 0);
            img.AlphaDataMapping = 'none';
            set(ax_heatmap, 'YDir', 'normal', 'Color', [1 1 1], ...
                'XLim', point_edges([1 end]), 'YLim', [0 1], ...
                'XTick', log10(1 + ticks), 'XTickLabel', string(ticks));
            xlabel(ax_heatmap, 'Number of points (log spacing)');
            ylabel(ax_heatmap, 'Confidence');
            hold(ax_heatmap, 'on');
            class_key = patch(ax_heatmap, NaN, NaN, class_color, 'EdgeColor', 'none', ...
                'DisplayName', sprintf('%s - %s (N=%d)', ...
                sensors{i}.name, class_names{j}, size(data,1)));
            legend(ax_heatmap, class_key, 'Location', 'northeast', 'Interpreter', 'none');
            grid(ax_heatmap, 'off');
            if isempty(data)
                text(ax_heatmap, mean(point_edges([1 end])), 0.5, 'No valid detections', ...
                    'HorizontalAlignment', 'center');
            end
        end
    end

    % Logarithmic colors keep sparse bins visible beside dense bins.
    set(ptcount_confidence_axes, 'ColorScale', 'log', 'CLim', [1 max(2, max_count)]);
    cb = colorbar(ptcount_confidence_axes(end));
    cb.Layout.Tile = 'east';
    cb.Label.String = 'Detection observations per bin (log scale)';
    cb.Label.Interpreter = 'none';
    linkaxes(ptcount_confidence_axes(:), 'xy');
end
