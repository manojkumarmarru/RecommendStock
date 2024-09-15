using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using StockPredictionAPI.Models;
using System;

public class HistoricalDataConverter : JsonConverter<HistoricalData>
{
    public override HistoricalData ReadJson(JsonReader reader, Type objectType, HistoricalData existingValue, bool hasExistingValue, JsonSerializer serializer)
    {
        JObject obj = JObject.Load(reader);
        var data = new HistoricalData
        {
            Date = (string)obj["date"],
            Open = (float)obj["open"],
            High = (float)obj["high"],
            Low = (float)obj["low"],
            Close = (float)obj["close"],
            Volume = (float)obj["volume"]
        };
        
        return data;
    }

    public override void WriteJson(JsonWriter writer, HistoricalData value, JsonSerializer serializer)
    {
        throw new NotImplementedException();
    }
}