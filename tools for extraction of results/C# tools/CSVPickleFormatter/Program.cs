using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace CSVPickleFormatter
{
    class Program
    {
        private static Conf conf = null;

        static void Main(string[] args)
        {
            try
            {
                // read configuration file
                conf = JsonSerializer.Deserialize<Conf>(File.ReadAllText("conf.json"));

                // get all .csv files
                var files = Directory.GetFiles(conf.ReadDirectory, "*.csv");

                // foreach file
                foreach (var fname in files)
                {
                    HandleFile(fname);

                    Console.WriteLine("");
                    Console.WriteLine("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }

            Console.WriteLine("Press key to terminate the program...");
            Console.Read();
        }

        private static void HandleFile(string fpath)
        {
            Console.WriteLine($"File handling: {fpath}");
            Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");

            // get filename
            var fname = Path.GetFileName(fpath).Replace(".csv", "");

            // create output directory
            var out_path = conf.WriteDirectory + fname;
            var info = Directory.CreateDirectory(out_path);
            Console.WriteLine($"> Creating output directory: {info.FullName}");

            // read content (lines)
            string[] lines = File.ReadAllLines(fpath, Encoding.UTF8);
            Console.WriteLine($"> Reading file: #lines ${lines.Length}");

            // line: string -> class
            var results = lines.Where(x => !x.Contains("step,al,learner,fold,ssl"))
                               .Select(x => new ExperimentalResult(x));

            // create file with new content                
            foreach (var byLearner in results.GroupBy(x => x.learner))
            {
                foreach (var byLearner_byAl in byLearner.GroupBy(x => x.al_strategy))
                {
                    foreach (var byLearner_byAl_bySsl in byLearner_byAl.GroupBy(x => x.ssl_strategy))
                    {
                        // content
                        var new_lines = byLearner_byAl_bySsl.Select(x => $"{x.iter},{x.step},{x.fold},{x.accuracy},{x.precesion},{x.recall},{x.f1score}").ToList();
                        new_lines.Insert(0, "iter,step,fold,acc,prec,rec,f1score");

                        // create the new file
                        var id = $"{byLearner.Key}$${byLearner_byAl.Key}$${byLearner_byAl_bySsl.Key}";
                        var out_fpath = out_path + $"\\{id}.csv";
                        CreateFile(out_fpath, new_lines);
                    }
                }
            }

            Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
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

    class ExperimentalResult
    {
        public string iter;
        public string step;
        public string fold;

        public string learner;
        public string al_strategy;
        public string ssl_strategy;

        public string accuracy;
        public string precesion;
        public string recall;
        public string f1score;

        public ExperimentalResult(string line)
        {
            var elms = line.Split(',');

            iter = elms[0];
            step = elms[1];
            al_strategy = elms[2];
            learner = elms[3];
            fold = elms[4];
            ssl_strategy = elms[5];
            accuracy = elms[6];
            precesion = elms[7];
            recall = elms[8];
            f1score = elms[9];
        }
    }
}
