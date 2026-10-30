function outStruct = normalize_stamps(inStruct)
    % Fase 1: Estrai tutti i valori dei campi che contengono 'stamp'
    values = collectStampValues(inStruct);
    values = values(values~=0);
    if isempty(values)
        outStruct = inStruct; % Nessun campo 'stamp', restituisci la struttura invariata
        return;
    end
    minVal = min(values);
    inStruct.time_offset_nsec = minVal*10^9;
    
    % Fase 2: Sottrai il minimo da tutti i campi 'stamp'
    outStruct = subtractMinFromStamps(inStruct, minVal);
end

function values = collectStampValues(s)
    values = [];
    fields = fieldnames(s);
    for i = 1:numel(fields)
        field = fields{i};
        value = s.(field);
        if isstruct(value)
            % Ricorsione per sottostrutture
            values = [values; collectStampValues(value)];
        elseif contains(lower(field), 'stamp') && isnumeric(value)
            % Aggiungi il valore se è numerico e il nome contiene 'stamp'
            values = [values; value(:)];
        end
    end
end

function sOut = subtractMinFromStamps(sIn, minVal)
    sOut = sIn;
    fields = fieldnames(sIn);
    for i = 1:numel(fields)
        field = fields{i};
        value = sIn.(field);
        if isstruct(value)
            % Ricorsione per sottostrutture
            sOut.(field) = subtractMinFromStamps(value, minVal);
        elseif contains(lower(field), 'stamp') && isnumeric(value)
            % Sottrai il minimo
            sOut.(field) = value - minVal;
        end
    end
end
