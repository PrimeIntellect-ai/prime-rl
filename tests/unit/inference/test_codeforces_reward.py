from zeroband.inference.genesys.codeforces import codeforces_reward

completion = (
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

verification_info = {
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

def test_codeforces_reward():
    score = codeforces_reward(completion, verification_info, verbose=False)
    assert score == (20 / 23)