from zeroband.inference.genesys.codeforces import codeforces_reward

# NOTE: Formatted prettily with pprint().

completion_stdio = (
 '#include <bits/stdc++.h>\n'
 'using namespace std;\n'
 '\n'
 'int main() {\n'
 '    ios_base::sync_with_stdio(false);\n'
 '    cin.tie(nullptr);\n'
 '    \n'
 '    int n, s;\n'
 '    cin >> n >> s;\n'
 '    \n'
 '    vector<int> a(n);\n'
 '    for (int i = 0; i < n; i++) {\n'
 '        cin >> a[i];\n'
 '    }\n'
 '    \n'
 '    // Check if already sorted\n'
 '    vector<int> sorted_a = a;\n'
 '    sort(sorted_a.begin(), sorted_a.end());\n'
 '    if (a == sorted_a) {\n'
 '        cout << 0 << "\\n";\n'
 '        return 0;\n'
 '    }\n'
 '    \n'
 '    // Build permutation that minimizes cycle count\n'
 '    // For each value, collect its positions in original and sorted arrays\n'
 '    map<int, vector<int>> value_to_positions;\n'
 '    map<int, vector<int>> value_to_sorted_positions;\n'
 '    \n'
 '    for (int i = 0; i < n; i++) {\n'
 '        value_to_positions[a[i]].push_back(i);\n'
 '    }\n'
 '    \n'
 '    for (int i = 0; i < n; i++) {\n'
 '        value_to_sorted_positions[sorted_a[i]].push_back(i);\n'
 '    }\n'
 '    \n'
 '    // Build permutation by matching positions optimally\n'
 '    vector<int> perm(n);\n'
 '    \n'
 '    for (auto& [val, positions] : value_to_positions) {\n'
 '        vector<int>& sorted_positions = value_to_sorted_positions[val];\n'
 '        \n'
 '        // For duplicates, use greedy matching to minimize cycles\n'
 '        // Try to keep elements in place when possible\n'
 '        vector<bool> used_orig(positions.size(), false);\n'
 '        vector<bool> used_sorted(sorted_positions.size(), false);\n'
 '        \n'
 '        // First pass: keep elements that are already in valid positions\n'
 '        for (int i = 0; i < positions.size(); i++) {\n'
 '            for (int j = 0; j < sorted_positions.size(); j++) {\n'
 '                if (!used_orig[i] && !used_sorted[j] && positions[i] == sorted_positions[j]) {\n'
 '                    perm[positions[i]] = sorted_positions[j];\n'
 '                    used_orig[i] = true;\n'
 '                    used_sorted[j] = true;\n'
 '                }\n'
 '            }\n'
 '        }\n'
 '        \n'
 '        // Second pass: assign remaining positions in order\n'
 '        int j = 0;\n'
 '        for (int i = 0; i < positions.size(); i++) {\n'
 '            if (!used_orig[i]) {\n'
 '                while (used_sorted[j]) j++;\n'
 '                perm[positions[i]] = sorted_positions[j];\n'
 '                used_sorted[j] = true;\n'
 '            }\n'
 '        }\n'
 '    }\n'
 '    \n'
 '    // Find all cycles\n'
 '    vector<vector<int>> cycles;\n'
 '    vector<bool> visited(n, false);\n'
 '    int total_length = 0;\n'
 '    \n'
 '    for (int i = 0; i < n; i++) {\n'
 '        if (!visited[i] && perm[i] != i) {\n'
 '            vector<int> cycle;\n'
 '            int curr = i;\n'
 '            \n'
 '            // Trace the cycle\n'
 '            while (!visited[curr]) {\n'
 '                visited[curr] = true;\n'
 '                cycle.push_back(curr);\n'
 '                curr = perm[curr];\n'
 '            }\n'
 '            \n'
 '            cycles.push_back(cycle);\n'
 '            total_length += cycle.size();\n'
 '        }\n'
 '    }\n'
 '    \n'
 '    // Check if total length exceeds s\n'
 '    if (total_length > s) {\n'
 '        cout << -1 << "\\n";\n'
 '        return 0;\n'
 '    }\n'
 '    \n'
 '    // Output the operations\n'
 '    cout << cycles.size() << "\\n";\n'
 '    for (const auto& cycle : cycles) {\n'
 '        cout << cycle.size() << "\\n";\n'
 '        for (int pos : cycle) {\n'
 '            cout << pos + 1 << " ";\n'
 '        }\n'
 '        cout << "\\n";\n'
 '    }\n'
 '    \n'
 '    return 0;\n'
 '}')

verification_info_stdio = {
    "language": "cpp",
    "input_mode": "stdio",
    "time_limit": 2.0,
    "memory_limit": 256.0,
    "tests": [
        {"input": "5 5\n3 2 3 1 1\n", "output": "1\n5\n1 4 2 3 5 \n"},
        {"input": "4 3\n2 1 4 3\n", "output": "-1"},
        {"input": "2 0\n2 2\n", "output": "0\n"},
        {"input": "1 0\n2\n", "output": "0\n"},
        {"input": "2 0\n2 1\n", "output": "-1"},

        {"input": "2 2\n2 1\n", "output": "1\n2\n1 2 \n"},
        {"input": "2 0\n1 1\n", "output": "0\n"},
        {"input": "2 1\n1 1\n", "output": "0\n"},
        {"input": "5 0\n1000000000 1000000000 1000000000 1000000000 1000000000\n", "output": "0\n"},
        {"input": "1 0\n258769137\n", "output": "0\n"},

        {"input": "5 0\n884430748 884430748 708433020 708433020 708433020\n", "output": "-1"},
        {"input": "5 4\n335381650 691981363 691981363 335381650 335381650\n", "output": "1\n4\n2 4 3 5 \n"},
        {"input": "5 2\n65390026 770505072 65390026 65390026 65390026\n", "output": "1\n2\n2 5 \n"},
        {"input": "5 200000\n682659092 302185582 518778252 29821187 14969298\n", "output": "2\n5\n1 5 2 3 4 \n2\n2 1 \n"},
        {"input": "5 4\n167616600 574805150 651016425 150949603 379708534\n", "output": "-1"},

        {"input": "5 5\n815605413 4894095 624809427 264202135 152952491\n", "output": "2\n3\n1 5 2 \n2\n3 4 \n"},
        {"input": "5 4\n201429826 845081337 219611799 598937628 680006294\n", "output": "1\n4\n2 5 4 3 \n"},
        {"input": "5 5\n472778319 561757623 989296065 99763286 352037329\n", "output": "1\n5\n1 3 5 2 4 \n"},
        {"input": "5 6\n971458729 608568364 891718769 464295315 98863653\n", "output": "2\n2\n1 5 \n3\n2 3 4 \n"},
        {"input": "5 4\n579487081 564229995 665920667 665920667 644707366\n", "output": "2\n2\n1 2 \n2\n3 5 \n"},

        {"input": "5 4\n81224924 319704343 319704343 210445208 128525140\n", "output": "1\n4\n2 4 3 5 \n"},
        {"input": "5 5\n641494999 641494999 228574099 535883079 535883079\n", "output": "1\n5\n1 4 2 5 3 \n"},
        {"input": "5 4\n812067558 674124159 106041640 106041640 674124159\n", "output": "-1"},
    ],
}

completion_file = (
 'import sys\n'
 'sys.setrecursionlimit(10**7)\n'
 '\n'
 'def is_winning(r, c, k, memo):\n'
 '    if (r, c) in memo:\n'
 '        return memo[(r, c)]\n'
 '    \n'
 '    # Base case: no moves available, current player loses\n'
 '    if r == 0 and c == 0:\n'
 '        return False\n'
 '    \n'
 '    # Try all possible moves\n'
 '    # If any move leads to a losing position, current position is winning\n'
 '    \n'
 '    # Move down (row - 1)\n'
 '    if r > 0:\n'
 '        if not is_winning(r-1, c, k, memo):\n'
 '            memo[(r, c)] = True\n'
 '            return True\n'
 '    \n'
 '    # Move right (col - 1)\n'
 '    if c > 0:\n'
 '        if not is_winning(r, c-1, k, memo):\n'
 '            memo[(r, c)] = True\n'
 '            return True\n'
 '    \n'
 '    # Move diagonally by k\n'
 '    if r >= k and c >= k:\n'
 '        if not is_winning(r-k, c-k, k, memo):\n'
 '            memo[(r, c)] = True\n'
 '            return True\n'
 '    \n'
 '    # All moves lead to winning positions, so current is losing\n'
 '    memo[(r, c)] = False\n'
 '    return False\n'
 '\n'
 '# Read input and process\n'
 "with open('input.txt', 'r') as f:\n"
 '    t, k = map(int, f.readline().split())\n'
 '    memo = {}\n'
 '    results = []\n'
 '    \n'
 '    for _ in range(t):\n'
 '        n, m = map(int, f.readline().split())\n'
 '        # From (1,1) to (n,m) means (n-1, m-1) moves needed\n'
 '        if is_winning(n-1, m-1, k, memo):\n'
 "            results.append('+')\n"
 '        else:\n'
 "            results.append('-')\n"
 '\n'
 '# Write output\n'
 "with open('output.txt', 'w') as f:\n"
 '    for result in results:\n'
 "        f.write(result + '\\n')\n")

verification_info_file = {
    'language': 'python',
    'input_mode': 'file',
    'time_limit': 2.0,
    'memory_limit': 64.0,
    'tests': [
        {'input': '10 2\n1 1\n1 2\n2 1\n2 2\n1 3\n2 3\n3 1\n3 2\n3 3\n4 3\n', 'output': '-\n+\n+\n-\n-\n+\n-\n+\n+\n+\n'},
        {'input': '20 2\n5 9\n6 7\n6 5\n9 5\n1 7\n7 5\n6 5\n2 10\n9 10\n5 5\n5 7\n3 3\n2 7\n6 1\n9 5\n1 1\n2 1\n5 8\n6 3\n2 9\n', 'output': '+\n+\n-\n+\n-\n+\n-\n-\n+\n+\n+\n+\n+\n+\n+\n-\n+\n-\n+\n+\n'},
        {'input': '20 3\n6 7\n9 10\n2 3\n4 1\n2 8\n8 2\n7 1\n8 9\n2 10\n1 3\n2 4\n8 6\n1 4\n4 8\n4 5\n10 5\n5 6\n1 4\n4 6\n1 5\n', 'output': '-\n+\n+\n+\n-\n-\n-\n+\n-\n-\n-\n+\n+\n+\n+\n-\n-\n+\n+\n-\n'},
        {'input': '20 5\n4 3\n6 6\n9 1\n2 9\n5 7\n9 6\n8 3\n8 5\n10 10\n9 2\n4 10\n8 5\n7 9\n10 1\n7 5\n3 4\n3 5\n7 6\n4 4\n3 5\n', 'output': '+\n+\n-\n+\n-\n+\n+\n+\n+\n+\n-\n+\n+\n+\n-\n+\n-\n+\n-\n-\n'},
        {'input': '20 7\n4 7\n4 3\n4 7\n10 10\n7 7\n10 7\n9 4\n10 2\n8 10\n10 10\n4 5\n8 2\n6 3\n5 2\n7 2\n4 3\n2 6\n1 9\n4 1\n2 6\n', 'output': '+\n+\n+\n+\n-\n+\n+\n-\n+\n+\n+\n-\n+\n+\n+\n+\n-\n-\n+\n-\n'},

        {'input': '20 10\n91 84\n66 30\n17 62\n82 43\n6 99\n44 17\n98 3\n81 95\n20 74\n75 22\n18 81\n14 26\n67 42\n69 3\n21 48\n36 27\n40 52\n38 65\n72 51\n74 64\n', 'output': '-\n-\n-\n-\n+\n-\n+\n+\n+\n+\n-\n+\n-\n-\n-\n+\n+\n-\n+\n+\n'},
        {'input': '20 29\n52 93\n66 67\n74 74\n16 42\n55 12\n50 23\n49 69\n32 29\n19 30\n70 3\n40 2\n46 84\n83 15\n40 46\n20 39\n29 88\n27 68\n81 51\n79 30\n17 89\n', 'output': '-\n+\n-\n-\n+\n+\n+\n+\n+\n+\n-\n+\n-\n+\n+\n+\n+\n+\n+\n-\n'},
        {'input': '20 55\n76 58\n61 97\n56 69\n10 56\n30 57\n57 4\n23 81\n42 26\n95 73\n70 33\n56 77\n35 15\n50 22\n11 18\n68 47\n18 70\n95 81\n46 12\n44 1\n92 13\n', 'output': '+\n+\n+\n-\n+\n+\n-\n-\n+\n+\n+\n-\n-\n+\n+\n-\n+\n-\n+\n+\n'},
        {'input': '20 101\n933 225\n664 100\n594 436\n885 717\n25 220\n221 400\n856 319\n896 112\n161 94\n117 40\n948 520\n351 191\n753 883\n911 199\n204 645\n301 295\n989 8\n268 144\n770 660\n186 756\n', 'output': '-\n-\n-\n+\n+\n+\n-\n+\n+\n+\n+\n+\n+\n+\n+\n-\n+\n+\n-\n+\n'},
        {'input': '20 221\n360 352\n978 849\n655 435\n882 729\n450 858\n301 419\n788 683\n916 906\n48 777\n531 937\n293 275\n912 823\n717 436\n139 22\n203 949\n557 629\n959 520\n350 709\n480 939\n431 464\n', 'output': '+\n-\n+\n-\n-\n+\n-\n-\n+\n-\n+\n-\n-\n+\n-\n-\n+\n-\n+\n-\n'},

        {'input': '20 17\n983 763\n179 855\n558 251\n262 553\n112 255\n500 844\n490 430\n398 373\n981 82\n121 341\n266 824\n690 77\n850 352\n144 637\n463 462\n21 917\n379 918\n114 847\n235 111\n367 331\n', 'output': '-\n+\n-\n+\n+\n+\n+\n+\n+\n-\n-\n+\n+\n+\n-\n+\n-\n+\n-\n-\n'},
        {'input': '20 39\n420 595\n253 306\n938 654\n79 40\n890 845\n36 302\n171 983\n805 833\n630 278\n199 589\n90 48\n807 416\n104 442\n20 682\n693 331\n864 758\n599 398\n180 282\n766 418\n968 122\n', 'output': '+\n+\n-\n+\n-\n-\n-\n-\n-\n-\n+\n+\n-\n-\n-\n-\n-\n-\n-\n+\n'},
        {'input': '20 1\n908 347\n933 924\n4 514\n190 177\n872 446\n250 800\n995 853\n241 124\n704 448\n921 24\n22 518\n496 531\n335 333\n683 470\n382 611\n886 707\n427 146\n673 175\n453 610\n411 112\n', 'output': '+\n+\n+\n+\n+\n+\n-\n+\n+\n+\n+\n+\n-\n+\n+\n+\n+\n-\n+\n+\n'},
        {'input': '20 67\n722 278\n207 851\n834 693\n367 673\n445 473\n43 432\n824 752\n587 411\n349 715\n611 232\n236 817\n57 699\n229 328\n197 574\n68 606\n864 882\n841 46\n2 311\n867 174\n860 7\n', 'output': '-\n+\n+\n+\n-\n+\n+\n-\n+\n-\n-\n-\n-\n+\n+\n-\n+\n+\n+\n+\n'},
        {'input': '20 4\n429 349\n827 701\n69 681\n375 716\n424 955\n238 741\n810 263\n665 754\n775 256\n165 864\n733 419\n607 956\n93 15\n894 929\n876 611\n436 559\n976 944\n10 261\n498 693\n360 199\n', 'output': '+\n-\n+\n+\n+\n-\n+\n+\n-\n+\n+\n-\n+\n+\n+\n-\n-\n+\n-\n-\n'},

        {'input': '20 9\n729 549\n316 664\n726 880\n153 633\n381 568\n689 228\n484 379\n804 236\n541 818\n553 596\n584 236\n459 683\n493 487\n795 243\n464 297\n705 547\n9 292\n24 756\n22 696\n510 344\n', 'output': '-\n+\n-\n+\n+\n+\n-\n+\n+\n-\n+\n+\n-\n-\n-\n-\n+\n-\n-\n-\n'},
        {'input': '20 16\n428 605\n221 378\n131 506\n649 969\n614 51\n593 985\n833 239\n876 646\n877 952\n272 180\n163 276\n642 873\n324 904\n296 238\n770 467\n167 851\n359 192\n877 722\n448 126\n850 379\n', 'output': '-\n+\n-\n-\n+\n-\n-\n+\n-\n-\n-\n-\n+\n+\n-\n+\n-\n+\n+\n+\n'},
        {'input': '20 6\n208 899\n423 886\n661 361\n616 94\n477 59\n348 936\n669 969\n732 606\n822 10\n309 757\n873 335\n348 906\n383 804\n584 784\n970 297\n155 343\n460 143\n686 670\n978 965\n209 257\n', 'output': '-\n+\n+\n+\n-\n+\n+\n-\n+\n-\n+\n+\n+\n+\n+\n-\n+\n+\n-\n+\n'},
        {'input': '20 827\n9 942\n611 876\n755 39\n93 115\n600 101\n547 228\n478 510\n629 770\n356 19\n647 936\n877 611\n975 973\n320 700\n29 467\n897 44\n738 977\n328 13\n774 448\n20 535\n186 90\n', 'output': '+\n+\n-\n-\n+\n+\n-\n+\n+\n+\n-\n+\n-\n-\n+\n+\n+\n-\n+\n-\n'},
        {'input': '20 1000\n178 171\n839 391\n192 334\n313 820\n25 214\n791 283\n354 984\n128 710\n516 670\n337 186\n952 416\n689 820\n471 410\n910 87\n35 188\n643 810\n846 407\n799 927\n637 138\n722 231\n', 'output': '+\n-\n-\n+\n+\n-\n-\n-\n-\n+\n-\n+\n+\n+\n+\n+\n+\n-\n+\n+\n'},
    ],
}

def test_codeforces_stdio_reward():
    stdio_score = codeforces_reward(completion_stdio, verification_info_stdio, verbose=False)
    assert stdio_score == (20 / 23)

def test_codeforces_file_reward():
    file_score = codeforces_reward(completion_file, verification_info_file, verbose=False)
    assert file_score == (20 / 20)