using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TSP1
{
    /*
     * ==============================================================================
     * Traveling Salesman Problem (TSP) Algorithm Implementation
     * ==============================================================================
     * 
     * [Problem Description and Complexity Explanation]
     * This program generates random points on a 2-D plane, and the goal is to find the "shortest closed loop distance" that connects all points and returns to the starting point.
     * - The "Decision Version" of the TSP (i.e., asking if there exists a loop with total distance <= a specific value) is a famous NP-Complete problem.
     * - The "find the absolute optimal shortest distance" that we solve here belongs to its Optimization Version, whose difficulty reaches NP-Hard.
     * - This program not only finds the shortest distance, but also lists all optimal paths that share this shortest distance.
     * 
     * [Data Structure]
     * - Each city (point) uses Int64 as coordinates, with a maximum value up to Int64.MaxValue / 1000.
     * - Distance uses Euclidean Distance. To avoid coordinate overflow, it converts to double, takes the square root, and rounds to the nearest integer.
     * - Each city is named directly as "{X}-{Y}".
     * 
     * [Algorithm Process]
     * 1. Initialize Distance Matrix and Pre-sorting:
     *    Calculate the distance between any two points, and build an array (_sortedEdges) for each point representing other points sorted from nearest to farthest.
     * 
     * 2. Greedy Initialization (Finding a good starting Upper Bound):
     *    - Find the two points A and B with the maximum distance in the entire graph.
     *    - Starting from A, use the "Nearest First (Nearest Neighbor)" strategy to gradually connect to the nearest unvisited point.
     *    - To force B to be the last point, temporarily exclude B in the Nearest First process.
     *    - After all other points are connected, finally connect to B, and then connect from B back to A, forming a valid loop.
     *    - This Greedy algorithm provides a decent initial _bestDistance for subsequent fast pruning.
     * 
     * [Algorithm Architecture and Academic Terms]
     * This program adopts the exact solving methods recognized as most efficient in academia, including the following core technologies:
     * 
     * 1. [Branch and Bound] and Depth-First Search (DFS):
     *    - Fix point A as the starting point, and pick branches from the unvisited points according to _sortedEdges (nearest first) to find a good upper bound as fast as possible.
     *    - [Important Pruning] When the current path total distance is already >= the known best solution (_bestDistance), immediately prune all subsequent branches.
     * 
     * 2. [1-Tree Lower Bound Pruning] - (Core concept of Concorde solver):
     *    - Top TSP solvers (such as Concorde) use this estimation mechanism. We implemented its core concept in the program:
     *    - Calculate the absolute shortest connection cost of remaining unvisited nodes using a [Minimum Spanning Tree (MST)] (Prim's algorithm).
     *    - If (current distance + MST remaining distance + minimum cost of connecting back to starting point) > known best solution, it means this branch can never break the record, and is terminated early.
     * 
     * 3. Parallel Design (Breadth-First Expansion for Perfect Load Balancing):
     *    - To maximize multi-core performance, the search tree is expanded for the first few levels to generate hundreds of independent tasks.
     *    - These tasks are then processed in parallel using Parallel.ForEach, ensuring no thread is idle and avoiding lock contention.
     * 
     * ==============================================================================
     */

    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            if (args.Length < 2)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("para1: 0, 1, 2 (0: brute force all possibilities, 1: prune to accelerate, 2: run both to verify results match)");
                Console.WriteLine("para2: Start N");
                Console.WriteLine("para3: (Optional) End N, if not provided, execute only start N");
                Console.WriteLine("para4: (Optional) Repeat count per test size, default is 1");
                Console.WriteLine("\nExample: TSP1.exe 1 10 12 1");
                return;
            }

            int mode = int.Parse(args[0]);
            int startN = int.Parse(args[1]);
            int endN = args.Length >= 3 ? int.Parse(args[2]) : startN;
            int repeatCount = args.Length >= 4 ? int.Parse(args[3]) : 1;

            TestRunner.RunAllTests(mode, startN, endN, repeatCount);
        }
    }

    public class Point
    {
        public int Id { get; set; }
        public long X { get; set; }
        public long Y { get; set; }
        public string Name => $"{X}-{Y}";
    }

    public class TSPSolver
    {
        private readonly List<Point> _points;
        private readonly int _n;
        private readonly long[][] _dist;
        private readonly int[][] _sortedEdges; // _sortedEdges[i] stores other point indexes sorted from nearest to farthest from i
        private readonly bool _enablePruning;

        private long _bestDistance = long.MaxValue;
        private readonly List<int[]> _allBestPaths = new List<int[]>();
        private readonly object _lock = new object();
        private long _combinationsChecked = 0;
        private readonly Action<string> _logger;

        public TSPSolver(List<Point> points, long[][] dist, bool enablePruning, Action<string>? logger = null)
        {
            _points = points;
            _n = points.Count;
            _dist = dist;
            _enablePruning = enablePruning;
            _logger = logger ?? (msg => Console.WriteLine(msg));

            // Pre-sort other points for each point (Nearest First)
            _sortedEdges = new int[_n][];
            for (int i = 0; i < _n; i++)
            {
                _sortedEdges[i] = Enumerable.Range(0, _n)
                     .Where(j => j != i)
                     .OrderBy(j => _dist[i][j])
                     .ToArray();
            }
        }

        private (long totalDist, int[] path) NearestFirstGreedySolve(int rootA, int rootB)
        {
            int[] greedyPath = new int[_n];
            bool[] greedyVisited = new bool[_n];
            greedyPath[0] = rootA;
            greedyVisited[rootA] = true;
            greedyVisited[rootB] = true; // Temporarily mark B as visited, so Nearest First doesn't pick it

            int curr = rootA;
            long greedyTotalDist = 0;
            for (int step = 1; step < _n - 1; step++)
            {
                int next = -1;
                long minDist = long.MaxValue;
                foreach (int v in _sortedEdges[curr])
                {
                    if (!greedyVisited[v] && _dist[curr][v] < minDist)
                    {
                        minDist = _dist[curr][v];
                        next = v;
                    }
                }
                greedyPath[step] = next;
                greedyVisited[next] = true;
                greedyTotalDist += minDist;
                curr = next;
            }

            // Finally connect to B, then back to A to form the loop
            greedyPath[_n - 1] = rootB;
            greedyTotalDist += _dist[curr][rootB];
            greedyTotalDist += _dist[rootB][rootA];

            return (greedyTotalDist, greedyPath);
        }

        public (long bestDist, List<int[]> bestPaths, long checkedCount) Solve()
        {
            // 1. Find the two points A, B with the maximum distance
            long maxDist = -1;
            int rootA = 0, rootB = 1;
            for (int i = 0; i < _n; i++)
            {
                for (int j = i + 1; j < _n; j++)
                {
                    if (_dist[i][j] > maxDist)
                    {
                        maxDist = _dist[i][j];
                        rootA = i;
                        rootB = j;
                    }
                }
            }

            // 2. Greedy algorithm to find initial solution (Greedy)
            // As requested: A is starting point, nearest first connects remaining points (excluding B), finally connects to B, then back to A
            var (greedyTotalDist, greedyPath) = NearestFirstGreedySolve(rootA, rootB);

            if (_enablePruning)
            {
                _bestDistance = greedyTotalDist;
                _allBestPaths.Clear();
                _allBestPaths.Add((int[])greedyPath.Clone());
            }
            else
            {
                _bestDistance = long.MaxValue;
            }

            // 3. Prepare parallel DFS Backtrack
            // Starting point is fixed to rootA, which avoids duplicate calculations from loop rotation
            if (_n <= 6)
            {
                // Small scale runs on single thread directly (no expansion needed)
                int[] path = new int[_n];
                path[0] = rootA;
                Backtrack(rootA, 1L << rootA, 0, 1, path);
            }
            else
            {
                // Breadth-First Expansion: Expand the DFS tree for the first two steps (A -> v1 -> v2)
                var tasks = new List<(int curr, long mask, long dist, int[] path)>();
                foreach (int v1 in _sortedEdges[rootA])
                {
                    foreach (int v2 in _sortedEdges[v1])
                    {
                        if (v2 == rootA) continue; // v2 cannot go back to A
                        
                        long mask = (1L << rootA) | (1L << v1) | (1L << v2);
                        long d = _dist[rootA][v1] + _dist[v1][v2];
                        int[] path = new int[_n];
                        path[0] = rootA; path[1] = v1; path[2] = v2;
                        tasks.Add((v2, mask, d, path));
                    }
                }

                // Sort tasks: Prioritize executing branches with the shortest first two steps (most likely to be optimal, strengthening subsequent pruning)
                tasks = tasks.OrderBy(t => t.dist).ToList();

                // Process in parallel using Parallel.ForEach (top-level Parallel.ForEach parallelized + inner purely synchronous)
                Parallel.ForEach(tasks, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, task =>
                {
                    // Each Task has an independent path array to avoid cross-thread pollution
                    int[] localPath = (int[])task.path.Clone();
                    Backtrack(task.curr, task.mask, task.dist, 3, localPath);
                });
            }

            return (_bestDistance, new List<int[]>(_allBestPaths), Interlocked.Read(ref _combinationsChecked));
        }

        private void Backtrack(int curr, long visitedMask, long currentDist, int visitedCount, int[] currentPath)
        {
            Interlocked.Increment(ref _combinationsChecked);

            // Lock-free fast path pruning (prune if greater than known best solution, keep equal for collecting equivalent best paths)
            long currentBest = Interlocked.Read(ref _bestDistance);
            //if (_enablePruning && currentDist > currentBest)
            //    return;

            // --- [1-Tree Lower Bound Pruning] ---
            // This is an advanced technique in Branch and Bound,
            // and the core concept used by world-class TSP solvers (like Concorde) to estimate the remaining lower bound.
            // By calculating the Minimum Spanning Tree (MST) of the remaining nodes using Prim's algorithm, 
            // it guarantees that the estimated remaining path cost is the absolute lowest bound.
            if (_enablePruning && visitedCount < _n)
            {
                long mstCost = 0;
                int startNode = currentPath[0];
                int firstUnvisited = -1;

                // Find the first unvisited node as the starting point for Prim's algorithm
                for (int i = 0; i < _n; i++)
                {
                    if ((visitedMask & (1L << i)) == 0)
                    {
                        firstUnvisited = i;
                        break;
                    }
                }

                if (firstUnvisited != -1)
                {
                    long mstMask = (1L << firstUnvisited);
                    // Used to record the minimum distance from MST to each unvisited node
                    long[] minEdgeToMst = new long[_n];
                    for (int i = 0; i < _n; i++)
                    {
                        if ((visitedMask & (1L << i)) == 0 && i != firstUnvisited)
                        {
                            minEdgeToMst[i] = _dist[firstUnvisited][i];
                        }
                    }

                    int remainingNodesToAdd = _n - visitedCount - 1; 
                    while (remainingNodesToAdd > 0)
                    {
                        long minEdge = long.MaxValue;
                        int nextNode = -1;
                        for (int i = 0; i < _n; i++)
                        {
                            if ((visitedMask & (1L << i)) == 0 && (mstMask & (1L << i)) == 0)
                            {
                                if (minEdgeToMst[i] < minEdge)
                                {
                                    minEdge = minEdgeToMst[i];
                                    nextNode = i;
                                }
                            }
                        }

                        mstCost += minEdge;
                        mstMask |= (1L << nextNode);
                        remainingNodesToAdd--;

                        // Update the minimum distance from nextNode (just added to MST) to other unvisited nodes not in MST
                        for (int i = 0; i < _n; i++)
                        {
                            if ((visitedMask & (1L << i)) == 0 && (mstMask & (1L << i)) == 0)
                            {
                                if (_dist[nextNode][i] < minEdgeToMst[i])
                                {
                                    minEdgeToMst[i] = _dist[nextNode][i];
                                }
                            }
                        }
                    }

                    // Calculate the minimum cost of connecting curr into the MST, and connecting MST back to startNode
                    long minE1 = long.MaxValue;
                    long minE2 = long.MaxValue;
                    for (int i = 0; i < _n; i++)
                    {
                        if ((visitedMask & (1L << i)) == 0)
                        {
                            if (_dist[curr][i] < minE1) minE1 = _dist[curr][i];
                            if (_dist[i][startNode] < minE2) minE2 = _dist[i][startNode];
                        }
                    }

                    long lowerBoundRest = mstCost + minE1 + minE2;
                    // If (distance already traveled + strict MST minimum remaining distance) > known best solution, it can never break the record, prune immediately!
                    if (currentDist + lowerBoundRest > currentBest)
                        return;
                }
            }

            // Visited all nodes, calculate distance back to the starting point
            if (visitedCount == _n)
            {
                long finalDist = currentDist + _dist[curr][currentPath[0]]; 
                
                // Lock-free check again
                if (finalDist <= currentBest)
                {
                    lock (_lock)
                    {
                        if (finalDist < _bestDistance)
                        {
                            _bestDistance = finalDist;
                            _allBestPaths.Clear();
                            _allBestPaths.Add((int[])currentPath.Clone());
                            int tid = Environment.CurrentManagedThreadId;
                            _logger($"[T:{tid:D2}]  >>> Better solution found! Total distance: {finalDist:N0} (best solution #1)");
                        }
                        else if (finalDist == _bestDistance)
                        {
                            _allBestPaths.Add((int[])currentPath.Clone());
                            int tid = Environment.CurrentManagedThreadId;
                            _logger($"[T:{tid:D2}]  >>> Equivalent best solution found! Total distance: {finalDist:N0} (best solution #{_allBestPaths.Count})");
                        }
                    }
                }
                return;
            }

            // Backtrack: Try all unvisited nodes sorted by nearest first
            foreach (int v in _sortedEdges[curr])
            {
                if ((visitedMask & (1L << v)) == 0)
                {
                    currentPath[visitedCount] = v;
                    Backtrack(v, visitedMask | (1L << v), currentDist + _dist[curr][v], visitedCount + 1, currentPath);
                }
            }
        }
    }

    public static class TestRunner
    {
        private static long NextLong(Random rng, long min, long max)
        {
            byte[] buf = new byte[8];
            rng.NextBytes(buf);
            long longRand = BitConverter.ToInt64(buf, 0);
            long range = max - min;
            if (range <= 0) return min; 
            return (Math.Abs(longRand % range)) + min;
        }

        public static void RunAllTests(int mode, int startN, int endN, int repeatCount)
        {
            var rng = new Random(42); // Fixed Seed for easier verification
            
            var testSizesList = new List<int>();
            for (int i = startN; i <= endN; i++) testSizesList.Add(i);
            int[] testSizes = testSizesList.ToArray();

            long maxCoord = Int64.MaxValue / 1000;
            string timeStamp = DateTime.Now.ToString("HHmmss");

            string outputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "output");
            outputDir = Path.GetFullPath(outputDir);
            if (!Directory.Exists(outputDir)) Directory.CreateDirectory(outputDir);

            var summaryLines = new List<string>
            {
                new string('═', 60),
                "  TSP (Traveling Salesman Problem) Execution Results Summary",
                new string('═', 60),
                "  The time complexity of Brute Force is O(N!)",
                "  As the number of cities N increases, the number of permutations grows factorially:"
            };

            Console.WriteLine("==================================================");
            Console.WriteLine(" TSP (Traveling Salesman Problem) Backtrack Search Test");
            Console.WriteLine("==================================================");

            foreach (int n in testSizes)
            {
                Console.WriteLine($"\n[ Generating Test Case N = {n} ]");
                BigInteger expectedCombinations = 0;
                if (n <= 6)
                {
                    for (int k = 0; k <= n - 1; k++)
                        expectedCombinations += Permutation(n - 1, k);
                }
                else
                {
                    for (int k = 2; k <= n - 1; k++)
                        expectedCombinations += Permutation(n - 1, k);
                }
                
                long totalCheckedPruned = 0, totalCheckedBrute = 0;
                TimeSpan totalTimePruned = TimeSpan.Zero, totalTimeBrute = TimeSpan.Zero;
                long maxCheckedPruned = long.MinValue, minCheckedPruned = long.MaxValue;
                long maxCheckedBrute = long.MinValue, minCheckedBrute = long.MaxValue;
                TimeSpan maxTimePruned = TimeSpan.MinValue, minTimePruned = TimeSpan.MaxValue;
                TimeSpan maxTimeBrute = TimeSpan.MinValue, minTimeBrute = TimeSpan.MaxValue;
                bool isMatch = true;

                StringBuilder sb = new StringBuilder();
                object sbLock = new object();
                Action<string> logger = msg => {
                    lock (sbLock) {
                        Console.WriteLine(msg);
                        sb.AppendLine(msg);
                    }
                };

                logger($"\n[ Generating Test Case N = {n} ]");
                logger("========================================");
                logger($" TSP Problem: N = {n}");
                logger("========================================");

                for (int rep = 0; rep < repeatCount; rep++)
                {
                    List<Point> points = new List<Point>();
                    for (int i = 0; i < n; i++)
                    {
                        points.Add(new Point
                        {
                            Id = i,
                            X = NextLong(rng, 0, maxCoord),
                            Y = NextLong(rng, 0, maxCoord)
                        });
                    }

                    long[][] dist = new long[n][];
                    for (int i = 0; i < n; i++)
                    {
                        dist[i] = new long[n];
                        for (int j = 0; j < n; j++)
                        {
                            if (i == j) dist[i][j] = 0;
                            else
                            {
                                double dx = (double)points[i].X - points[j].X;
                                double dy = (double)points[i].Y - points[j].Y;
                                dist[i][j] = (long)Math.Round(Math.Sqrt(dx * dx + dy * dy));
                            }
                        }
                    }

                    logger($"\n[ Rep {rep + 1}/{repeatCount} ]");
                    
                    long distBrute = -1, distPruned = -1;

                    if (mode == 0 || mode == 2)
                    {
                        TSPSolver solverBrute = new TSPSolver(points, dist, false, logger);
                        var swBrute = Stopwatch.StartNew();
                        var resBrute = solverBrute.Solve();
                        swBrute.Stop();
                        
                        totalCheckedBrute += resBrute.checkedCount;
                        totalTimeBrute += swBrute.Elapsed;
                        if (resBrute.checkedCount > maxCheckedBrute) maxCheckedBrute = resBrute.checkedCount;
                        if (resBrute.checkedCount < minCheckedBrute) minCheckedBrute = resBrute.checkedCount;
                        if (swBrute.Elapsed > maxTimeBrute) maxTimeBrute = swBrute.Elapsed;
                        if (swBrute.Elapsed < minTimeBrute) minTimeBrute = swBrute.Elapsed;
                        distBrute = resBrute.bestDist;
                        
                        logger($"\n[Best Solution] [Brute Force]");
                        logger($"  Min total distance: {distBrute:N0}");
                        logger($"  Found {resBrute.bestPaths.Count} solution(s)");
                        logger($"  Actual combinations checked: {resBrute.checkedCount:N0}");
                        logger($"  Execution time: {swBrute.Elapsed.TotalMilliseconds:F2} ms");
                        for (int k = 0; k < resBrute.bestPaths.Count; k++)
                        {
                            string pStr = string.Join(" -> ", resBrute.bestPaths[k].Select(idx => points[idx].Name));
                            logger($"    Solution #{k + 1}: {pStr} -> {points[resBrute.bestPaths[k][0]].Name}");
                        }
                    }

                    if (mode == 1 || mode == 2)
                    {
                        TSPSolver solverPruned = new TSPSolver(points, dist, true, logger);
                        var swPruned = Stopwatch.StartNew();
                        var resPruned = solverPruned.Solve();
                        swPruned.Stop();
                        
                        totalCheckedPruned += resPruned.checkedCount;
                        totalTimePruned += swPruned.Elapsed;
                        if (resPruned.checkedCount > maxCheckedPruned) maxCheckedPruned = resPruned.checkedCount;
                        if (resPruned.checkedCount < minCheckedPruned) minCheckedPruned = resPruned.checkedCount;
                        if (swPruned.Elapsed > maxTimePruned) maxTimePruned = swPruned.Elapsed;
                        if (swPruned.Elapsed < minTimePruned) minTimePruned = swPruned.Elapsed;
                        distPruned = resPruned.bestDist;
                        
                        logger($"\n[Best Solution] [Pruning]");
                        logger($"  Min total distance: {distPruned:N0}");
                        logger($"  Found {resPruned.bestPaths.Count} solution(s)");
                        logger($"  Actual combinations checked: {resPruned.checkedCount:N0}");
                        logger($"  Execution time: {swPruned.Elapsed.TotalMilliseconds:F2} ms");
                        for (int k = 0; k < resPruned.bestPaths.Count; k++)
                        {
                            string pStr = string.Join(" -> ", resPruned.bestPaths[k].Select(idx => points[idx].Name));
                            logger($"    Solution #{k + 1}: {pStr} -> {points[resPruned.bestPaths[k][0]].Name}");
                        }
                    }

                    if (mode == 2 && distBrute != distPruned)
                    {
                        isMatch = false;
                        logger($"  ❌ Mismatch! Brute: {distBrute:N0}, Pruned: {distPruned:N0}");
                    }
                    
                    if (repeatCount > 1 && rep < repeatCount - 1)
                    {
                        logger(new string('!', 60));
                    }
                }

                long avgCheckedPruned = repeatCount > 0 ? totalCheckedPruned / repeatCount : 0;
                long avgCheckedBrute = repeatCount > 0 ? totalCheckedBrute / repeatCount : 0;
                TimeSpan avgTimePruned = TimeSpan.FromTicks(repeatCount > 0 ? totalTimePruned.Ticks / repeatCount : 0);
                TimeSpan avgTimeBrute = TimeSpan.FromTicks(repeatCount > 0 ? totalTimeBrute.Ticks / repeatCount : 0);
                if (repeatCount == 0) {
                    maxCheckedPruned = 0; minCheckedPruned = 0; maxCheckedBrute = 0; minCheckedBrute = 0;
                    maxTimePruned = TimeSpan.Zero; minTimePruned = TimeSpan.Zero; maxTimeBrute = TimeSpan.Zero; minTimeBrute = TimeSpan.Zero;
                }

                logger(new string('=', 60));
                logger($"[Summary] N={n}, executed {repeatCount} time(s)");
                if (mode == 0 || mode == 2)
                {
                    logger($"  [Brute Force] Actual combinations checked (average): {avgCheckedBrute:N0}");
                    if (repeatCount > 1) {
                        logger($"  [Brute Force] Actual combinations checked (max): {maxCheckedBrute:N0}");
                        logger($"  [Brute Force] Actual combinations checked (min): {minCheckedBrute:N0}");
                    }
                    logger($"  [Brute Force] Execution time (average): {avgTimeBrute.TotalMilliseconds:F2} ms");
                    if (repeatCount > 1) {
                        logger($"  [Brute Force] Execution time (max): {maxTimeBrute.TotalMilliseconds:F2} ms");
                        logger($"  [Brute Force] Execution time (min): {minTimeBrute.TotalMilliseconds:F2} ms");
                    }
                }
                if (mode == 1 || mode == 2)
                {
                    logger($"  [Pruning] Actual combinations checked (average): {avgCheckedPruned:N0}");
                    if (repeatCount > 1) {
                        logger($"  [Pruning] Actual combinations checked (max): {maxCheckedPruned:N0}");
                        logger($"  [Pruning] Actual combinations checked (min): {minCheckedPruned:N0}");
                    }
                    logger($"  [Pruning] Execution time (average): {avgTimePruned.TotalMilliseconds:F2} ms");
                    if (repeatCount > 1) {
                        logger($"  [Pruning] Execution time (max): {maxTimePruned.TotalMilliseconds:F2} ms");
                        logger($"  [Pruning] Execution time (min): {minTimePruned.TotalMilliseconds:F2} ms");
                    }
                }
                logger(new string('=', 60));

                string outFilePath = Path.Combine(outputDir, $"tsp{n}-{timeStamp}.txt");
                File.WriteAllText(outFilePath, sb.ToString());

                summaryLines.Add("");
                summaryLines.Add($"  [ N = {n} ]");
                summaryLines.Add($"    Expected max combinations : {expectedCombinations:N0}");

                if (mode == 0 || mode == 2)
                {
                    summaryLines.Add($"    [Brute Force]");
                    summaryLines.Add($"      Actual combinations checked (average) : {avgCheckedBrute:N0}");
                    if (repeatCount > 1) {
                        summaryLines.Add($"      Actual combinations checked (max) : {maxCheckedBrute:N0}");
                        summaryLines.Add($"      Actual combinations checked (min) : {minCheckedBrute:N0}");
                    }
                    summaryLines.Add($"      Execution time (average)       : {avgTimeBrute.TotalMilliseconds:F2} ms");
                    if (repeatCount > 1) {
                        summaryLines.Add($"      Execution time (max)       : {maxTimeBrute.TotalMilliseconds:F2} ms");
                        summaryLines.Add($"      Execution time (min)       : {minTimeBrute.TotalMilliseconds:F2} ms");
                    }
                }
                if (mode == 1 || mode == 2)
                {
                    summaryLines.Add($"    [Pruning]");
                    summaryLines.Add($"      Actual combinations checked (average) : {avgCheckedPruned:N0}");
                    if (repeatCount > 1) {
                        summaryLines.Add($"      Actual combinations checked (max) : {maxCheckedPruned:N0}");
                        summaryLines.Add($"      Actual combinations checked (min) : {minCheckedPruned:N0}");
                    }
                    double pruneRatio = avgCheckedPruned > 0 ? (double)expectedCombinations / avgCheckedPruned : 0;
                    summaryLines.Add($"      Speedup ratio (average)   : {pruneRatio:F2}x (Expected / Actual)");
                    summaryLines.Add($"      Execution time (average)       : {avgTimePruned.TotalMilliseconds:F2} ms");
                    if (repeatCount > 1) {
                        summaryLines.Add($"      Execution time (max)       : {maxTimePruned.TotalMilliseconds:F2} ms");
                        summaryLines.Add($"      Execution time (min)       : {minTimePruned.TotalMilliseconds:F2} ms");
                    }
                }
                if (mode == 2)
                {
                    summaryLines.Add($"    => Mismatch between the two modes: {(isMatch ? "No" : "Yes")}");
                }
            }

            string summaryFilePath = Path.Combine(outputDir, $"tspSummary-{timeStamp}.txt");
            File.WriteAllLines(summaryFilePath, summaryLines);
            Console.WriteLine($"\n✅ All results exported to: {outputDir}");
        }

        private static BigInteger Permutation(int n, int k)
        {
            if (k < 0 || k > n) return 0;
            if (k == 0) return 1;
            BigInteger res = 1;
            for (int i = 0; i < k; i++)
            {
                res *= (n - i);
            }
            return res;
        }
    }
}
