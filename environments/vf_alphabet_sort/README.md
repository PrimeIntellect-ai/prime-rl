# vf-alphabet-sort

This task requires the model to maintain and update an alphabetically sorted list of names across multiple conversation turns, with new names being tagged appropriately. The dataset uses real author names from arXiv papers, with 1-3 turns per conversation and 2-5 total names (the turn and name counts are randomized during the data creation process by default).

The reward function uses difflib to calculate sequence similarity between predicted and expected outputs, with the final score raised to the nth power (similarity_power, defaults to 4) to emphasize precision.

## Example Input/ Outputs

### Turn 1

**Input**

```txt
Sort these names in alphabetical order by FIRST name: Johnson, Alice, Bob, Charlie

Use exactly this format:
<alphabetical_sorted>
Name1
Name2
Name3
Name4
</alphabetical_sorted>
```

**Output**

```txt
<alphabetical_sorted>
Alice
Bob
Charlie
Johnson
</alphabetical_sorted>
```

### Turn 2

**Input**

```txt
Now sort ALL of these names alphabetically by FIRST name: David, Emily

These are in addition to the prior list. Mark any NEW names (that weren't in the prior list) with `// new name!` at the end.

Use exactly this format:
<combined_alphabetical_sorted>
Name1
Name2
Name3 // new name!
Name4
Name5 // new name!
Name6
</combined_alphabetical_sorted>
```

**Output**

```txt
<combined_alphabetical_sorted>
Alice
Bob
Charlie
David // new name!
Emily // new name!
Johnson
</combined_alphabetical_sorted>
```

### Turn 3

```txt
Now sort ALL of these names alphabetically by FIRST name: Frank

These are in addition to the prior list. Mark any NEW names (that weren't in the prior list) with `// new name!` at the end. Follow the same format as before.
```

```txt
<combined_alphabetical_sorted>
Alice
Bob
Charlie
David
Emily
Frank // new name!
Johnson
</combined_alphabetical_sorted>
```

