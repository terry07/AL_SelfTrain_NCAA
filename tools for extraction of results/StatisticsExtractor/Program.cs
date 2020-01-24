using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace StatisticsExtractor
{
    class Program
    {
        private static Conf conf = null;

        static void Main(string[] args)
        {
            // read configuration file
            conf = JsonSerializer.Deserialize<Conf>(File.ReadAllText("conf.json"));

            // get subfolders
            DirectoryInfo[] subfolders = new DirectoryInfo(conf.ReadDirectory).GetDirectories();

            // for each subfolder
            foreach (var folder in subfolders)
            {
                Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");

                // create output directory
                var out_path = conf.WriteDirectory + folder.Name;
                var info = Directory.CreateDirectory(out_path);
                Console.WriteLine($"> Creating output directory: {info.FullName}");

                // get all .csv files
                var files = folder.GetFiles("*.csv");

                // foreach file
                foreach (var finfo in files)
                {
                    ExtractStatistics(out_path, finfo);

                    Console.WriteLine("");
                    Console.WriteLine("");
                }
            }
        }

        private static void ExtractStatistics(string out_path, FileInfo finfo)
        {
            var in_fpath = finfo.FullName;
            var out_fpath = $"{out_path}\\{finfo.Name}".Replace(".csv", "");

            Console.WriteLine($"> File handling: {in_fpath}");

            // read content (lines)
            string[] lines = File.ReadAllLines(in_fpath, Encoding.UTF8);
            Console.WriteLine($"> Reading file: #lines ${lines.Length}");

            // line: string -> class
            var results = lines.Where(x => !x.Contains("iter,step,fold"))
                               .Select(x => new Line(x));

            // extract improvement
            ExtractImprovement(out_fpath + "_improvement.csv", results);

            // extract stability
            ExtractStability(out_fpath + "_stability.csv", results);

            // extract average
            ExtractAverage(out_fpath + "_avg.csv", results);

            // extract std
            ExtractStandardDeviation(out_fpath + "_std.csv", results);

            Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
        }

        private static void ExtractImprovement(string fpath, IEnumerable<Line> results)
        {
            var new_lines = new List<string>();

            // create file with new content
            foreach (var bystep in results.GroupBy(x => x.step))
            {
                foreach (var bystep_byfold in bystep.GroupBy(x => x.fold))
                {
                    // get start/end iteration
                    var temp = bystep_byfold.OrderBy(x => int.Parse(x.iter));
                    var iter_start = temp.First();
                    var iter_end = temp.Last();

                    // content
                    new_lines.Add($"{bystep.Key}," +
                                  $"{bystep_byfold.Key}," +
                                  $"{float.Parse(iter_end.accuracy) - float.Parse(iter_start.accuracy)}," +
                                  $"{float.Parse(iter_end.precesion) - float.Parse(iter_start.precesion)}," +
                                  $"{float.Parse(iter_end.recall) - float.Parse(iter_start.recall)}," +
                                  $"{float.Parse(iter_end.f1score) - float.Parse(iter_start.f1score)}");
                }
            }

            new_lines.Insert(0, "step,fold,acc,prec,rec,f1score");

            // create the new file                    
            CreateFile(fpath, new_lines);
        }

        private static void ExtractStability(string fpath, IEnumerable<Line> results)
        {
            var new_lines = new List<string>();

            // create file with new content
            foreach (var bystep in results.GroupBy(x => x.step))
            {
                foreach (var bystep_byfold in bystep.GroupBy(x => x.fold))
                {
                    var iterations = bystep_byfold.OrderBy(x => int.Parse(x.iter));

                    var min_value = iterations.First();
                    var count_acc = 0;
                    var count_prec = 0;
                    var count_rec = 0;
                    var count_f1score = 0;
                    for (int i = 1; i < iterations.Count(); i++)
                    {
                        if (float.Parse(iterations.ElementAt(i).accuracy) < float.Parse(min_value.accuracy)) { count_acc++; }
                        if (float.Parse(iterations.ElementAt(i).precesion) < float.Parse(min_value.precesion)) { count_prec++; }
                        if (float.Parse(iterations.ElementAt(i).recall) < float.Parse(min_value.recall)) { count_rec++; }
                        if (float.Parse(iterations.ElementAt(i).f1score) < float.Parse(min_value.f1score)) { count_f1score++; }

                        min_value = iterations.ElementAt(i);
                    }

                    // content
                    new_lines.Add($"{bystep.Key}," +
                                    $"{bystep_byfold.Key}," +
                                    $"{Math.Round(count_acc * 100f / iterations.Count(), 2)}," +
                                    $"{Math.Round(count_prec * 100f / iterations.Count(), 2)}," +
                                    $"{Math.Round(count_rec * 100f / iterations.Count(), 2)}," +
                                    $"{Math.Round(count_f1score * 100f / iterations.Count(), 2)}");
                }
            }

            new_lines.Insert(0, "step,fold,acc,prec,rec,f1score");

            // create the new file                    
            CreateFile(fpath, new_lines);
        }

        private static void ExtractAverage(string fpath, IEnumerable<Line> results)
        {
            var new_lines = new List<string>();

            // create file with new content
            foreach (var bystep in results.GroupBy(x => x.step))
            {
                // get last iteration info
                var folds = new List<Line>();
                foreach (var bystep_byfold in bystep.GroupBy(x => x.fold))
                {
                    folds.Add(bystep_byfold.OrderBy(x => int.Parse(x.iter)).Last());
                }

                // content
                new_lines.Add($"{bystep.Key}," +
                              $"{folds.Select(x => float.Parse(x.accuracy)).Average()}," +
                              $"{folds.Select(x => float.Parse(x.precesion)).Average()}," +
                              $"{folds.Select(x => float.Parse(x.recall)).Average()}," +
                              $"{folds.Select(x => float.Parse(x.f1score)).Average()}");
            }

            new_lines.Insert(0, "step,acc,prec,rec,f1score");

            // create the new file                    
            CreateFile(fpath, new_lines);
        }

        public static void ExtractStandardDeviation(string fpath, IEnumerable<Line> results)
        {
            var new_lines = new List<string>();

            // create file with new content
            foreach (var bystep in results.GroupBy(x => x.step))
            {
                // get last iteration info
                var folds = new List<Line>();
                foreach (var bystep_byfold in bystep.GroupBy(x => x.fold))
                {
                    folds.Add(bystep_byfold.OrderBy(x => int.Parse(x.iter)).Last());
                }

                // content
                new_lines.Add($"{bystep.Key}," +
                              $"{StdDev(folds.Select(x => float.Parse(x.accuracy)))}," +
                              $"{StdDev(folds.Select(x => float.Parse(x.precesion)))}," +
                              $"{StdDev(folds.Select(x => float.Parse(x.recall)))}," +
                              $"{StdDev(folds.Select(x => float.Parse(x.f1score)))}");
            }

            new_lines.Insert(0, "step,acc,prec,rec,f1score");

            // create the new file                    
            CreateFile(fpath, new_lines);
        }

        private static double StdDev(IEnumerable<float> values, bool as_sample = false)
        {
            // Get the mean.
            double mean = values.Average();

            // Get the sum of the squares of the differences
            // between the values and the mean.
            var squares_query = values.Select(x => (x - mean) * (x - mean));
            double sum_of_squares = squares_query.Sum();

            if (as_sample)
            {
                return Math.Sqrt(sum_of_squares / (values.Count() - 1));
            }
            else
            {
                return Math.Sqrt(sum_of_squares / values.Count());
            }
        }

        private static void CreateFile(string fpath, IEnumerable<string> lines)
        {
            Console.WriteLine($"> Creating new file (#lines {lines.Count()}): {Path.GetFileName(fpath)}");
            File.WriteAllLines(fpath, lines);
        }
    }

    public class Conf
    {
        public string ReadDirectory { get; set; }
        public string WriteDirectory { get; set; }
    }

    public class Line
    {
        public string iter;
        public string step;
        public string fold;
        public string accuracy;
        public string precesion;
        public string recall;
        public string f1score;

        public Line(string line)
        {
            var elms = line.Split(',');

            iter = elms[0];
            step = elms[1];
            fold = elms[2];
            accuracy = elms[3];
            precesion = elms[4];
            recall = elms[5];
            f1score = elms[6];
        }
    }
}
