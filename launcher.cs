using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace GeoSRStudio
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Title = "GeoSR Studio - Satellite Super-Resolution (SRS v1.2)";
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("===============================================================================");
            Console.WriteLine("            GeoSR Studio: Satellite Image Super-Resolution System");
            Console.WriteLine("       SRS v1.2 Compliant | Hamdard University & SUPARCO FYP Project");
            Console.WriteLine("===============================================================================");
            Console.ResetColor();
            Console.WriteLine();

            // Set current directory to application root
            string appDir = AppDomain.CurrentDomain.BaseDirectory;
            Directory.SetCurrentDirectory(appDir);

            // 1. Locate Python
            Console.WriteLine("[1/3] Detecting Python & Deep Learning Runtime...");
            string pythonPath = FindPython();

            if (string.IsNullOrEmpty(pythonPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n[ERROR] Python runtime was not found on your system!");
                Console.WriteLine("Please install Anaconda or Python 3.10+ and add it to your PATH.");
                Console.ResetColor();
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" -> Found Python: " + pythonPath);
            Console.ResetColor();

            // 2. Check and Auto-Install Dependencies
            Console.WriteLine("\n[2/3] Verifying Required Libraries (torch, streamlit, rasterio, etc.)...");
            bool depsOk = CheckDependencies(pythonPath);

            if (!depsOk)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(" -> Missing packages detected! Automatically installing required libraries...");
                Console.WriteLine("    Please wait while packages are installed via pip...");
                Console.ResetColor();
                Console.WriteLine();

                bool installOk = InstallDependencies(pythonPath);
                if (!installOk)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("\n[WARNING] Could not install all packages automatically.");
                    Console.WriteLine("Attempting to launch application anyway...");
                    Console.ResetColor();
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("\n[OK] All required packages installed successfully!");
                    Console.ResetColor();
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine(" -> All required packages are already installed and verified!");
                Console.ResetColor();
            }

            // 3. Launch Streamlit Application
            Console.WriteLine("\n[3/3] Starting GeoSR Studio Web Dashboard...");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(" -> Local Server URL: http://localhost:8501");
            Console.WriteLine(" -> Launching your default web browser automatically...");
            Console.ResetColor();
            Console.WriteLine("\n-------------------------------------------------------------------------------");
            Console.WriteLine("  Press Ctrl+C in this window at any time to stop the server.");
            Console.WriteLine("-------------------------------------------------------------------------------\n");

            // Open browser after a short delay
            new Thread(() =>
            {
                Thread.Sleep(2500);
                try
                {
                    Process.Start(new ProcessStartInfo("http://localhost:8501") { UseShellExecute = true });
                }
                catch { }
            }).Start();

            // Run streamlit
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pythonPath;
            psi.Arguments = "-m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false";
            psi.UseShellExecute = false;
            psi.WorkingDirectory = appDir;

            try
            {
                using (Process proc = Process.Start(psi))
                {
                    proc.WaitForExit();
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n[ERROR] Failed to start application: " + ex.Message);
                Console.ResetColor();
                Console.WriteLine("\nPress any key to exit...");
                Console.ReadKey();
            }
        }

        static string FindPython()
        {
            // Check direct known paths
            string[] knownPaths = new string[]
            {
                @"D:\ANACONDA\python.exe",
                @"C:\ProgramData\anaconda3\python.exe",
                @"C:\Users\" + Environment.UserName + @"\anaconda3\python.exe",
                @"C:\Users\" + Environment.UserName + @"\AppData\Local\Programs\Python\Python312\python.exe",
                @"C:\Users\" + Environment.UserName + @"\AppData\Local\Programs\Python\Python311\python.exe",
                @"C:\Users\" + Environment.UserName + @"\AppData\Local\Programs\Python\Python310\python.exe",
                @"C:\Python312\python.exe",
                @"C:\Python311\python.exe",
                @"C:\Python310\python.exe"
            };

            foreach (string p in knownPaths)
            {
                if (File.Exists(p)) return p;
            }

            // Check PATH
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo("python.exe", "--version");
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                using (Process p = Process.Start(psi))
                {
                    p.WaitForExit();
                    if (p.ExitCode == 0) return "python.exe";
                }
            }
            catch { }

            return null;
        }

        static bool CheckDependencies(string pythonPath)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonPath;
                psi.Arguments = "-c \"import torch, streamlit, rasterio, PIL, skimage, safetensors, matplotlib\"";
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                using (Process p = Process.Start(psi))
                {
                    p.WaitForExit();
                    return (p.ExitCode == 0);
                }
            }
            catch
            {
                return false;
            }
        }

        static bool InstallDependencies(string pythonPath)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonPath;
                psi.Arguments = "-m pip install -r requirements.txt";
                psi.UseShellExecute = false;

                using (Process p = Process.Start(psi))
                {
                    p.WaitForExit();
                    return (p.ExitCode == 0);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error during pip install: " + ex.Message);
                return false;
            }
        }
    }
}
