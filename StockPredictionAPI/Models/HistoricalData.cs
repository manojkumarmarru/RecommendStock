namespace StockPredictionAPI.Models
{
    using Microsoft.ML.Data;

    public class HistoricalData
    {
        public string Date { get; set; }
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public float Volume { get; set; }
        [ColumnName("Label")]
        public float Label { get; set; }
    }
}