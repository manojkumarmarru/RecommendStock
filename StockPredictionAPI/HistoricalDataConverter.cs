using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using StockPredictionAPI.Models;
using System;

public class HistoricalDataConverter : JsonConverter
{
    public override bool CanConvert(Type objectType)
    {
        return objectType == typeof(HistoricalData);
    }

    public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
    {
        var jsonObject = JObject.Load(reader);
        var historicalData = new HistoricalData
        {
            Date = (string)jsonObject["date"],
            Open = (float)jsonObject["open"],
            High = (float)jsonObject["high"],
            Low = (float)jsonObject["low"],
            Close = (float)jsonObject["close"],
            Volume = (float)jsonObject["volume"]
        };
        return historicalData;
    }

    public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
    {
        throw new NotImplementedException();
    }
}