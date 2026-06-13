using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;

/*
 * ═════════════════════════════════════════════════════════════════════════
 *  [Academic Background and Complexity Explanation]
 * ═════════════════════════════════════════════════════════════════════════
 *  The problem solved by this program is the "Atypical Knapsack Problem", which is essentially:
 *  1. [An optimization variant of the Subset Sum Problem]
 *  2. [The Value-free 0/1 Knapsack Problem (only size/weight is considered)]
 *
 *  ■ Problem Definition:
 *    Given a set of items S = {s_1, s_2, ..., s_n}, where each item has only a single attribute "Size",
 *    and a knapsack "Capacity limit" C.
 *    Our goal is to find a subset S' ⊆ S such that:
 *      - The total size ∑_{s_i ∈ S'} size(s_i) <= C
 *      - And the total size is maximized (closest to the capacity limit C).
 *
 *  ■ NP-Complete / NP-Hard Properties:
 *    - The decision version of the traditional Subset Sum problem (i.e., asking if there exists a subset whose sum is exactly equal to C)
 *      is one of Karp's 21 famous NP-Complete problems.
 *    - The "total size maximization" we are solving is its Optimization Version.
 *      Since we can directly use the solution of the optimization problem to answer the decision problem, this problem is NP-Hard.
 *    - Under the brute-force search method, we must enumerate all 2^n subset possibilities. As the number of items n grows exponentially,
 *      the computation time faces a "combinatorial explosion", which is a classic characteristic of NP-Complete/NP-Hard problems.
 *
 *  ■ Algorithm Architecture and Academic Terms (KnapsackP2 Version):
 *    This program adopts the following top-tier algorithmic techniques:
 *    1. [Greedy Algorithm]: Obtain a greedy initial solution GreedySolve() from largest to smallest to get a strong initial Lower Bound.
 *    2. [Branch and Bound]: Call BacktrackLoop, enumerating from the last item to the first item using DFS.
 *    3. [Suffix Sum Optimization]:
 *       Paired with powerful pruning logic (PruneA and PruneB), using pre-calculated remaining sums,
 *       it quickly obtains the Upper Bound in O(1) complexity and significantly reduces the search space.
 *       - PruneA (Feasibility Check): Pruning based on feasibility.
 *       - PruneB (Bounding by Remaining Sum): Pruning based on remaining upper bound.
 * ═════════════════════════════════════════════════════════════════════════
 */

namespace KnapsackP2
{
    public class Item
    {
        public string Name { get; set; }
        public long Size { get; set; }

        public Item(string name, long size)
        {
            Name = name;
            Size = size;
        }

        public override string ToString() => $"{Name}(Size={Size})";
    }

    public class BacktrackKnapsackSolver
    {
        public static readonly long MinValueLimit = 1;
        public static readonly long MaxValueLimit = long.MaxValue / 1000;
        //public static readonly long MaxValueLimit = 1000000;

        public static readonly bool small2big = false;
        //public static readonly bool small2big = true;

        private long _bestSize;
        private bool[] _bestSelection;
        private long _bestRemaining;
        private readonly object _lock = new object();
        private long _combinationsChecked;

        private readonly List<bool[]> _allBestSelections = new List<bool[]>();
        private readonly SemaphoreSlim _threadSemaphore = new SemaphoreSlim(15, 15);

        private readonly List<Item> _items;
        private readonly long _capacity;
        private readonly KnapsackLogger _logger;
        private readonly bool _verbose;
        private readonly bool _enablePruning;

        private readonly long[] _pruneA1Counts;
        private readonly long[] _pruneB1Counts;
        private readonly long[] _pruneA2Counts;
        private readonly long[] _pruneB2Counts;

        public BacktrackKnapsackSolver(List<Item> items, long capacity, KnapsackLogger logger, bool verbose, bool enablePruning)
        {
            _items = items;
            _capacity = capacity;
            _logger = logger;
            _verbose = verbose;
            _enablePruning = enablePruning;
            _bestSize = 0;
            _bestSelection = new bool[items.Count];
            _bestRemaining = capacity;
            _combinationsChecked = 0;
            _pruneA1Counts = new long[items.Count];
            _pruneB1Counts = new long[items.Count];
            _pruneA2Counts = new long[items.Count];
            _pruneB2Counts = new long[items.Count];
        }

        public (long bestSize, List<Item> bestItems, long checkedCount, long[] pruneA1, long[] pruneA2, long[] pruneB1, long[] pruneB2) Solve()
        {
            int n = _items.Count;
            long totalCombinations = 1L << n;

#region hide
            LogHelper.PrintSolveStart(_logger, n, _capacity, totalCombinations, _enablePruning);
#endregion

            // 1. Get greedy initial solution
            bool[] greedySelection = new bool[n];
            long greedyRemaining = GreedySolve(greedySelection);

#region hide
            LogHelper.PrintGreedyResult(_logger, _items, greedySelection, _capacity - greedyRemaining, greedyRemaining);
            if (_verbose)
            {
                int tid = Environment.CurrentManagedThreadId;
                _logger.Log($"[T:{tid:D2}]  ── Greedy initial solution completed: Remaining {greedyRemaining:N0}");
            }
#endregion

            // Evaluate greedy initial combination
            CheckAndUpdate(greedySelection, greedyRemaining);

            // Determine depth d for parallel expansion
            int d = Math.Min(8, n); // Expand first 8 items, generating 2^8 = 256 independent subtrees
            if (n <= 20) d = 0; // Small scale does not need parallel execution

            if (d == 0)
            {
                // 2. Start single-threaded Backtrack loop
                BacktrackLoop(greedySelection, 0, greedyRemaining);
            }
            else
            {
                // Generate 2^d independent initial states
                int numStates = 1 << d;
                var validStates = new List<(bool[] state, long remaining)>();

                for (int i = 0; i < numStates; i++)
                {
                    bool[] state = (bool[])greedySelection.Clone();
                    long remaining = _capacity;

                    // Determine include/exclude of first d items based on bits of i
                    for (int bit = 0; bit < d; bit++)
                    {
                        bool isIn = ((i >> bit) & 1) == 1;
                        state[bit] = isIn;
                        if (isIn)
                        {
                            remaining -= _items[bit].Size;
                        }
                    }

                    if (remaining >= 0)
                    {
                        validStates.Add((state, remaining));
                    }
                }

                // Sort valid states from smallest to largest remaining space (smaller remaining means fuller, most likely to be the best solution, executing first strengthens pruning)
                validStates.Sort((a, b) => a.remaining.CompareTo(b.remaining));

                // Process these hundreds of independent tasks in parallel, with purely synchronous DFS internally to avoid deadlocks!
                Parallel.ForEach(validStates, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, st =>
                {
                    CheckAndUpdate(st.state, st.remaining);
                    BacktrackLoop(st.state, d, st.remaining);
                });
            }

#region hide
            long checkedCount = Interlocked.Read(ref _combinationsChecked);
            long totalCombinations2 = 1L << n;
            LogHelper.PrintSolveEnd(_logger, checkedCount, totalCombinations2, _pruneA1Counts, _pruneA2Counts, _pruneB1Counts, _pruneB2Counts);
#endregion

            List<Item> bestItems = new List<Item>();
            long bestSize;
            List<bool[]> allBestCopy;
            lock (_lock)
            {
                bestSize = _bestSize;
                for (int i = 0; i < n; i++)
                {
                    if (_bestSelection[i])
                        bestItems.Add(_items[i]);
                }
                allBestCopy = new List<bool[]>(_allBestSelections);
            }

#region hide
            LogHelper.PrintAllBestSolutions(_logger, _items, allBestCopy, bestSize, _capacity);
#endregion

            return (bestSize, bestItems, checkedCount, _pruneA1Counts, _pruneA2Counts, _pruneB1Counts, _pruneB2Counts);
        }

        /// <summary>
        /// Greedy initial solution: Try placing items from largest to smallest, modify selection, return remaining space.
        /// </summary>
        private long GreedySolve(bool[] selection)
        {
            long remaining = _capacity;
            for (int i = 0; i < _items.Count; i++)
            {
                if (_items[i].Size <= remaining)
                {
                    selection[i] = true;
                    remaining -= _items[i].Size;
                }
                else
                {
                    selection[i] = false;
                }
            }
            return remaining;
        }

        private void BacktrackLoop(bool[] selection, int startIdx, long currentRemaining)
        {
            int n = _items.Count;
            if (small2big)
            {
                for (int i = n - 1; i >= startIdx; i--)
                {
                    ProcessBacktrackBranch(selection, startIdx, i, currentRemaining);
                }
            }
            else
            {
                for (int i = startIdx; i < n; i++)
                {
                    ProcessBacktrackBranch(selection, startIdx, i, currentRemaining);
                }
            }
        }

        /// <summary>
        /// Process a single backtrack branch: Flip item i, evaluate the new combination, and recursively process subsequent i+1~N items.
        /// Uses three-level pruning of Branch and Bound, paired with Suffix Sum Optimization:
        /// - PruneA (Feasibility Check): Feasibility pruning (even if we include all remaining items after removing one, we cannot match the known best solution).
        /// - PruneB (Upper Bound Pruning / Bounding by Remaining Sum): Remaining upper bound pruning (exceeds capacity even if we try to adjust).
        /// - PruneC: Upper bound is lower than the known best solution.
        /// </summary>
        private void ProcessBacktrackBranch(bool[] selection, int startIdx, int flipIdx, long parentRemaining)
        {
            // Flip the flipIdx-th item
            bool[] flipped = (bool[])selection.Clone();
            bool isNowIn = !selection[flipIdx];
            flipped[flipIdx] = isNowIn;

            // Recalculate the overall remaining space
            long currentSizeDelta = isNowIn ? -_items[flipIdx].Size : _items[flipIdx].Size;
            long currentRemaining = parentRemaining + currentSizeDelta;

            if (_enablePruning)
            {
                // Calculate the sum of out-of-bag and in-bag items for subsequent items first (calculated in a single loop)
                long sumOutAfter = 0;
                long sumInAfter = 0;
                for (int k = flipIdx + 1; k < _items.Count; k++)
                {
                    if (flipped[k])
                        sumInAfter += _items[k].Size;
                    else
                        sumOutAfter += _items[k].Size;
                }

                if (!isNowIn) // IN -> OUT (remaining space increases)
                {
                    // PruneA1: Subsequent out-of-bag sum < size of the removed item -> cannot make up for it, definitely worse than the parent node
                    if (sumOutAfter < _items[flipIdx].Size)
                    {
                        Interlocked.Increment(ref _pruneA1Counts[flipIdx]);
                        CheckAndUpdate(flipped, currentRemaining);
#region hide
                        if (_verbose)
                        {
                            int tid = Environment.CurrentManagedThreadId;
                            _logger.Log($"[T:{tid:D2}]  ✂ PruneA1@{startIdx} Item {_items[flipIdx].Name}: In -> Out, subsequent out-of-bag sum {sumOutAfter:N0} < removed size {_items[flipIdx].Size:N0}");
                        }
#endregion
                        return;
                    }

                    // PruneA2: Current remaining space - subsequent out-of-bag sum >= current best remaining space (if current best remaining space is 0, then > 0)
                    long bestRemainingSnapshot = Interlocked.Read(ref _bestRemaining);
                    long bestPossibleRemaining = currentRemaining - sumOutAfter;
                    bool shouldPruneA2 = false;

                    if (bestRemainingSnapshot == 0)
                    {
                        if (bestPossibleRemaining > 0) shouldPruneA2 = true;
                    }
                    else
                    {
                        if (bestPossibleRemaining >= bestRemainingSnapshot) shouldPruneA2 = true;
                    }

                    if (shouldPruneA2)
                    {
                        Interlocked.Increment(ref _pruneA2Counts[flipIdx]);
                        CheckAndUpdate(flipped, currentRemaining);
#region hide
                        if (_verbose)
                        {
                            int tid = Environment.CurrentManagedThreadId;
                            _logger.Log($"[T:{tid:D2}]  ✂ PruneA2@{startIdx} Item {_items[flipIdx].Name}: Best possible remaining {bestPossibleRemaining:N0} >= known best remaining {bestRemainingSnapshot:N0}");
                        }
#endregion
                        return;
                    }
                }
                else // OUT -> IN (remaining space decreases)
                {
                    // PruneB1: Even if we remove all subsequent in-bag items, the remaining space is still negative -> cannot be valid
                    if (currentRemaining + sumInAfter < 0)
                    {
                        Interlocked.Increment(ref _pruneB1Counts[flipIdx]);
                        CheckAndUpdate(flipped, currentRemaining);
#region hide
                        if (_verbose)
                        {
                            int tid = Environment.CurrentManagedThreadId;
                            _logger.Log($"[T:{tid:D2}]  ✂ PruneB1@{startIdx} Item {_items[flipIdx].Name}: Out -> In, new remaining {currentRemaining:N0} + removable {sumInAfter:N0} < 0");
                        }
#endregion
                        return;
                    }

                    // PruneB2: Current remaining space - subsequent out-of-bag sum >= current best remaining space (if current best remaining space is 0, then > 0)
                    long bestRemainingSnapshot = Interlocked.Read(ref _bestRemaining);
                    long bestPossibleRemaining = currentRemaining - sumOutAfter;
                    bool shouldPruneB2 = false;

                    if (bestRemainingSnapshot == 0)
                    {
                        if (bestPossibleRemaining > 0) shouldPruneB2 = true;
                    }
                    else
                    {
                        if (bestPossibleRemaining >= bestRemainingSnapshot) shouldPruneB2 = true;
                    }

                    if (shouldPruneB2)
                    {
                        Interlocked.Increment(ref _pruneB2Counts[flipIdx]);
                        CheckAndUpdate(flipped, currentRemaining);
#region hide
                        if (_verbose)
                        {
                            int tid = Environment.CurrentManagedThreadId;
                            _logger.Log($"[T:{tid:D2}]  ✂ PruneB2@{startIdx} Item {_items[flipIdx].Name}: Best possible remaining {bestPossibleRemaining:N0} >= known best remaining {bestRemainingSnapshot:N0}");
                        }
#endregion
                        return;
                    }
                }
            }

            // Evaluate and update the current best solution
            CheckAndUpdate(flipped, currentRemaining);

#region hide
            if (_verbose)
            {
                int tid = Environment.CurrentManagedThreadId;
                string action = isNowIn ? "Out -> In" : "In -> Out";
                _logger.Log($"[T:{tid:D2}]  Backtrack@{startIdx} Item {_items[flipIdx].Name}: {action}, Remaining space: {currentRemaining:N0}");
            }
#endregion

            // Recursively backtrack on i+1 ~ N
            int nextIdx = flipIdx + 1;
            if (nextIdx < _items.Count)
            {
                BacktrackLoop(flipped, nextIdx, currentRemaining);
            }
        }

        /// <summary>
        /// Check a selection plan, and update if it is valid and better than the current best solution.
        /// </summary>
        private void CheckAndUpdate(bool[] selection, long remaining)
        {
            Interlocked.Increment(ref _combinationsChecked);

            if (remaining < 0)
                return; // Invalid, exceeds capacity limit

            long currentSize = _capacity - remaining;

            // Lock-free fast path: only try to enter lock when current solution >= historical best solution (solves severe Lock Contention)
            if (currentSize < _bestSize)
                return;

            lock (_lock)
            {
                if (currentSize > _bestSize)
                {
                    _bestSize = currentSize;
                    _bestRemaining = remaining;
                    Array.Copy(selection, _bestSelection, selection.Length);
                    _allBestSelections.Clear();
                    _allBestSelections.Add((bool[])selection.Clone());

#region hide
                    LogHelper.PrintBetterSolutionFound(_logger, _verbose, _bestSize, _bestRemaining, _capacity, _allBestSelections.Count);
#endregion
                }
                else if (currentSize == _bestSize && currentSize > 0)
                {
                    // Only record and display equivalent best solutions when the remaining space is 0 (perfect solution)
                    if (remaining == 0)
                    {
                        _allBestSelections.Add((bool[])selection.Clone());

#region hide
                        LogHelper.PrintEqualBestSolutionFound(_logger, _verbose, currentSize, remaining, _allBestSelections.Count);
#endregion
                    }
                }
            }
        }

        public static (long bestSize, List<Item> bestItems, long checkedCount, long[] pruneA1, long[] pruneA2, long[] pruneB1, long[] pruneB2) Solve(
            List<Item> items, long capacity, KnapsackLogger logger, bool verbose = false, bool enablePruning = true)
        {
            var solver = new BacktrackKnapsackSolver(items, capacity, logger, verbose, enablePruning);
            return solver.Solve();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            if (args.Length < 2)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("para1: 0 or 1, whether to prune (0: brute force all possibilities, 1: prune to accelerate)");
                Console.WriteLine("para2: Start N");
                Console.WriteLine("para3: (Optional) End N, if not provided, execute only start N");
                Console.WriteLine("para4: (Optional) Repeat count per test size, default is 1");
                Console.WriteLine("\nExample: KnapsackP2.exe 1 20 25 1");
                return;
            }

            bool enablePruning = int.Parse(args[0]) == 1;
            int startN = int.Parse(args[1]);
            int endN = args.Length >= 3 ? int.Parse(args[2]) : startN;
            int repeatCount = args.Length >= 4 ? int.Parse(args[3]) : 1;

            TestRunner.RunAllTests(enablePruning, startN, endN, repeatCount);
        }
    }

    #region Test execution and logger utility tools

    public static class TestRunner
    {
        private static long NextLong(Random rng, long min, long max)
        {
            byte[] buf = new byte[8];
            rng.NextBytes(buf);
            long longVal = BitConverter.ToInt64(buf, 0);
            return Math.Abs(longVal % (max - min)) + min;
        }

        private static (List<Item> sortedItems, long capacity, string answersStr) GenerateProblem(Random rng, int count)
        {
            var rawItems = new List<Item>();
            for (int i = 1; i <= count; i++)
            {
                rawItems.Add(new Item($"Item{i:D2}", NextLong(rng, BacktrackKnapsackSolver.MinValueLimit, BacktrackKnapsackSolver.MaxValueLimit)));
            }

            var sortedItems = rawItems.OrderByDescending(x => x.Size).ToList();

            long capacity = 0;
            var answers = new List<string>();
            foreach (var item in sortedItems)
            {
                bool isInside = rng.Next(0, 2) == 1;
                if (isInside)
                {
                    capacity += item.Size;
                    answers.Add($"{item.Name}(In)");
                }
                else
                {
                    answers.Add($"{item.Name}(Out)");
                }
            }

            string answersStr = string.Join(", ", answers);
            return (sortedItems, capacity, answersStr);
        }

        public static void RunAllTests(bool enablePruning, int startN, int endN, int repeatCount)
        {
            var rng = new Random(); // Remove fixed random seed to make each run a completely fresh random case
            string sharedTimeStamp = DateTime.Now.ToString("HHmmss");

            var testSizesList = new List<int>();
            for (int i = startN; i <= endN; i++)
            {
                testSizesList.Add(i);
            }
            int[] testSizes = testSizesList.ToArray();

            var results = new List<(int N, long Expected, long AvgActual, long MaxActual, long MinActual, TimeSpan AvgElapsed, TimeSpan MaxElapsed, TimeSpan MinElapsed, int RepeatCount)>();

            for (int t = 0; t < testSizes.Length; t++)
            {
                int n = testSizes[t];
                long combos = 1L << n;
                string comboStr = combos.ToString("N0");

                using (var logger = new KnapsackLogger(n, sharedTimeStamp))
                {
                    long totalActualCombos = 0;
                    TimeSpan totalElapsed = TimeSpan.Zero;
                    long maxActualCombos = long.MinValue;
                    long minActualCombos = long.MaxValue;
                    TimeSpan maxElapsed = TimeSpan.MinValue;
                    TimeSpan minElapsed = TimeSpan.MaxValue;

                    for (int rep = 0; rep < repeatCount; rep++)
                    {
                        string desc = $"{n} items, {comboStr} combinations, size type is Int64 (KnapsackP2 version)";
                        if (repeatCount > 1) desc += $" (Repetition {rep + 1}/{repeatCount})";
                        if (n >= 25)
                            desc += "\n  ⚠ Warning: This may take several seconds to several minutes of execution time";

                        LogHelper.PrintTestHeader(logger, t + 1, desc);

                        var (items, capacity, answersStr) = GenerateProblem(rng, n);

#region hide
                        LogHelper.PrintGeneratedItems(logger, items);
                        LogHelper.PrintSolutionSetup(logger, answersStr, capacity);
#endregion

                        var sw = Stopwatch.StartNew();
                        bool verbose = (n <= 10); // Enable verbose for cases where N <= 10
                        
                        var (bestSize, bestItems, actualCombos, pruneA1, pruneA2, pruneB1, pruneB2) = BacktrackKnapsackSolver.Solve(items, capacity, logger, verbose, enablePruning);
                        sw.Stop();

                        LogHelper.PrintResult(logger, bestSize, bestItems, capacity, sw.Elapsed, actualCombos);
                        
                        totalActualCombos += actualCombos;
                        totalElapsed += sw.Elapsed;
                        if (actualCombos > maxActualCombos) maxActualCombos = actualCombos;
                        if (actualCombos < minActualCombos) minActualCombos = actualCombos;
                        if (sw.Elapsed > maxElapsed) maxElapsed = sw.Elapsed;
                        if (sw.Elapsed < minElapsed) minElapsed = sw.Elapsed;

                        if (repeatCount > 1 && rep < repeatCount - 1)
                        {
                            logger.Log(new string('!', 60));
                        }
                    }

                    long avgCombos = repeatCount > 0 ? totalActualCombos / repeatCount : 0;
                    TimeSpan avgElapsed = TimeSpan.FromTicks(repeatCount > 0 ? totalElapsed.Ticks / repeatCount : 0);
                    if (repeatCount == 0) { maxActualCombos = 0; minActualCombos = 0; maxElapsed = TimeSpan.Zero; minElapsed = TimeSpan.Zero; }

                    logger.Log(new string('=', 60));
                    logger.Log($"[Summary] N={n}, executed {repeatCount} time(s)");
                    logger.Log($"  Actual combinations checked (average): {avgCombos:N0}");
                    if (repeatCount > 1) {
                        logger.Log($"  Actual combinations checked (max): {maxActualCombos:N0}");
                        logger.Log($"  Actual combinations checked (min): {minActualCombos:N0}");
                    }
                    logger.Log($"  Execution time (average): {avgElapsed.TotalMilliseconds:F2} ms");
                    if (repeatCount > 1) {
                        logger.Log($"  Execution time (max): {maxElapsed.TotalMilliseconds:F2} ms");
                        logger.Log($"  Execution time (min): {minElapsed.TotalMilliseconds:F2} ms");
                    }
                    logger.Log(new string('=', 60));

                    results.Add((n, combos, avgCombos, maxActualCombos, minActualCombos, avgElapsed, maxElapsed, minElapsed, repeatCount));
                }
            }

            LogHelper.PrintSummaryToFile(sharedTimeStamp, results);
        }
    }

    public class KnapsackLogger : IDisposable
    {
        private readonly StreamWriter _fileWriter;
        private readonly string _filePath;
        private readonly object _logLock = new object();

        public static string GetOutputDir()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string projectDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", ".."));
            string outputDir = Path.Combine(projectDir, "output");
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);
            return outputDir;
        }

        public KnapsackLogger(int n, string sharedTimeStamp)
        {
            string fileName = $"knapsack{n}-{sharedTimeStamp}.txt";
            string outputDir = GetOutputDir();

            _filePath = Path.Combine(outputDir, fileName);
            _fileWriter = new StreamWriter(_filePath, false, System.Text.Encoding.UTF8) { AutoFlush = true };

            Log($"[Logger Initialized] Log file will be written to: {_filePath}");
            Log(new string('═', 60));
        }

        public void Log(string message)
        {
            lock (_logLock)
            {
                Console.WriteLine(message);
                _fileWriter.WriteLine(message);
            }
        }

        public void Dispose()
        {
            _fileWriter?.Dispose();
        }
    }

    public static class LogHelper
    {
        public static void PrintTestHeader(KnapsackLogger logger, int testCaseNum, string description)
        {
            logger.Log("\n" + new string('═', 60));
            logger.Log($"  Test Case {testCaseNum}: {description}");
            logger.Log(new string('═', 60));
        }

        public static void PrintGeneratedItems(KnapsackLogger logger, List<Item> items)
        {
            logger.Log("Generated item list (sorted from largest to smallest):");
            logger.Log(string.Join(", ", items));
            logger.Log(new string('─', 60));
        }

        public static void PrintSolutionSetup(KnapsackLogger logger, string answersStr, long capacity)
        {
            logger.Log("Default random solution (In/Out):");
            logger.Log(answersStr);
            logger.Log($"Derived knapsack capacity limit (which is the total size of guaranteed feasible solution): {capacity:N0}");
            logger.Log(new string('─', 60));
        }

        public static void PrintSolveStart(KnapsackLogger logger, int n, long capacity, long totalCombinations, bool enablePruning)
        {
            logger.Log($"Item count: {n}");
            logger.Log($"Knapsack capacity limit (total size): {capacity:N0}");
            logger.Log($"Total combinations: {totalCombinations:N0}");
            logger.Log($"Smart pruning: {(enablePruning ? "ON" : "OFF")}");
            logger.Log(new string('─', 60));
        }

        public static void PrintSolveEnd(KnapsackLogger logger, long checkedCount, long expectedCount, long[] pruneA1, long[] pruneA2, long[] pruneB1, long[] pruneB2)
        {
            logger.Log(new string('─', 60));
            logger.Log($"Combinations checked: {checkedCount:N0}");
            logger.Log($"Expected combinations (2^N): {expectedCount:N0}");
            
            long totalPruneA1 = pruneA1.Sum();
            long totalPruneA2 = pruneA2.Sum();
            long totalPruneB1 = pruneB1.Sum();
            long totalPruneB2 = pruneB2.Sum();
            
            if (totalPruneA1 > 0 || totalPruneA2 > 0 || totalPruneB1 > 0 || totalPruneB2 > 0)
            {
                logger.Log($"  ✂ Pruning statistics: PruneA1 = {totalPruneA1:N0} times, PruneA2 = {totalPruneA2:N0} times, PruneB1 = {totalPruneB1:N0} times, PruneB2 = {totalPruneB2:N0} times");
                for (int i = 0; i < pruneA1.Length; i++)
                {
                    if (pruneA1[i] > 0 || pruneA2[i] > 0 || pruneB1[i] > 0 || pruneB2[i] > 0)
                        logger.Log($"     Item {i,2}: PruneA1 = {pruneA1[i],8}, PruneA2 = {pruneA2[i],8}, PruneB1 = {pruneB1[i],8}, PruneB2 = {pruneB2[i],8}");
                }
            }

            if (checkedCount == expectedCount)
            {
                logger.Log($"  ✔ Combination counts match, successfully enumerated all 2^N combinations.");
            }
            else
            {
                long diff = expectedCount - checkedCount;
                long pruneTotal = totalPruneA1 + totalPruneA2 + totalPruneB1 + totalPruneB2;
                logger.Log($"  ✂ Pruning saved {diff:N0} branch expansions (PruneA1+A2+B1+B2 triggered {pruneTotal:N0} times)");
            }
        }

        public static void PrintGreedyResult(KnapsackLogger logger, List<Item> items, bool[] selection, long totalSize, long remaining)
        {
            logger.Log("── Greedy Initial Solution ──");
            var inItems = new List<string>();
            var outItems = new List<string>();
            for (int i = 0; i < items.Count; i++)
            {
                if (selection[i])
                    inItems.Add(items[i].ToString());
                else
                    outItems.Add(items[i].ToString());
            }
            logger.Log($"  In-bag: {string.Join(", ", inItems)}");
            logger.Log($"  Out-of-bag: {string.Join(", ", outItems)}");
            logger.Log($"  Total size: {totalSize:N0}, Remaining space: {remaining:N0}");
            logger.Log(new string('─', 60));
        }

        public static void PrintBetterSolutionFound(KnapsackLogger logger, bool verbose, long bestSize, long bestRemaining, long capacity, int solutionCount)
        {
            int tid = Environment.CurrentManagedThreadId;
            string suffix = bestRemaining == 0 ? $" (best solution #{solutionCount})" : "";
            logger.Log($"[T:{tid:D2}]  >>> Better solution found! Total size: {bestSize:N0}, Remaining space: {bestRemaining:N0}{suffix}");
        }

        public static void PrintEqualBestSolutionFound(KnapsackLogger logger, bool verbose, long size, long remaining, int solutionCount)
        {
            int tid = Environment.CurrentManagedThreadId;
            logger.Log($"[T:{tid:D2}]  >>> Equivalent best solution found! Total size: {size:N0}, Remaining space: {remaining:N0} (best solution #{solutionCount})");
        }

        public static void PrintAllBestSolutions(KnapsackLogger logger, List<Item> items, List<bool[]> allBestSelections, long bestSize, long capacity)
        {
            long bestRemaining = capacity - bestSize;
            logger.Log($"\n[All Best Solutions List] Total of {allBestSelections.Count} set(s) (Total size: {bestSize:N0}, Remaining space: {bestRemaining:N0})");
            for (int s = 0; s < allBestSelections.Count; s++)
            {
                var sel = allBestSelections[s];
                var inItems = new List<string>();
                for (int i = 0; i < items.Count; i++)
                {
                    if (sel[i])
                        inItems.Add(items[i].ToString());
                }
                logger.Log($"  Solution #{s + 1}: {string.Join(", ", inItems)}");
            }
        }

        public static void PrintResult(KnapsackLogger logger, long bestSize, List<Item> bestItems, long capacity, TimeSpan elapsed, long actualCombos)
        {
            long remaining = capacity - bestSize;
            logger.Log("\n[Best Solution]");
            logger.Log($"  Max total size: {bestSize:N0}");
            logger.Log($"  Remaining space: {remaining:N0}");
            logger.Log($"  Selected items: {string.Join(", ", bestItems)}");
            logger.Log($"  Actual combinations checked: {actualCombos:N0}");
            logger.Log($"  Execution time: {elapsed.TotalMilliseconds:F2} ms");
        }

        public static void PrintSummaryToFile(string sharedTimeStamp, List<(int N, long Expected, long AvgActual, long MaxActual, long MinActual, TimeSpan AvgElapsed, TimeSpan MaxElapsed, TimeSpan MinElapsed, int RepeatCount)> results)
        {
            string outputDir = KnapsackLogger.GetOutputDir();
            string summaryPath = Path.Combine(outputDir, $"knapsackSummary-{sharedTimeStamp}.txt");

            var lines = new List<string>
            {
                new string('═', 60),
                "  NP-Complete Properties Summary and Execution Results (KnapsackP2)",
                new string('═', 60),
                "  The time complexity of Brute Force is O(2^n * n)",
                "  As the number of items n increases, the number of combinations grows exponentially:"
            };

            foreach (var r in results)
            {
                lines.Add("");
                lines.Add($"  [ N = {r.N} ]");
                lines.Add($"    Expected combinations (2^N) : {r.Expected:N0}");
                lines.Add($"    Actual combinations checked (average) : {r.AvgActual:N0}");
                if (r.RepeatCount > 1) {
                    lines.Add($"    Actual combinations checked (max) : {r.MaxActual:N0}");
                    lines.Add($"    Actual combinations checked (min) : {r.MinActual:N0}");
                }
                
                double pruneRatio = r.AvgActual > 0 ? (double)r.Expected / r.AvgActual : 0;
                lines.Add($"    Speedup ratio (average)   : {pruneRatio:F2}x (Expected / Actual)");
                
                lines.Add($"    Execution time (average)       : {r.AvgElapsed.TotalMilliseconds:F2} ms");
                if (r.RepeatCount > 1) {
                    lines.Add($"    Execution time (max)       : {r.MaxElapsed.TotalMilliseconds:F2} ms");
                    lines.Add($"    Execution time (min)       : {r.MinElapsed.TotalMilliseconds:F2} ms");
                }
            }

            lines.Add("");
            lines.Add("    n=38  -> Over 274.8 billion combinations");
            lines.Add(new string('═', 60));

            using (var sw = new StreamWriter(summaryPath, false, System.Text.Encoding.UTF8))
            {
                foreach (var line in lines)
                {
                    Console.WriteLine(line);
                    sw.WriteLine(line);
                }
            }
            Console.WriteLine($"[Summary] Written to: {summaryPath}");
        }
    }
    #endregion
}
