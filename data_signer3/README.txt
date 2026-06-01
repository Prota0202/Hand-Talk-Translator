Signeur 3 — adulte (cross-signeur)
==================================

Protocole (7 glosses, sans alphabet) :
  MOI, NOM, Bonjour, 2, 4, ans, ETUDIANT
  5 echantillons par gloss | mode AUTO

Deja collecte : MOI, NOM, Bonjour, 2, 4
Reste a faire  : ans, ETUDIANT

Reprendre la collecte :
  .\scripts\collect_signer3.ps1

Ou manuellement :
  python collect_data.py --data-dir data_signer3 --signer3 --resume --samples 5 --auto

Apres la collecte :
  .\scripts\eval_cross_signers.ps1

Le modele n'est PAS reentraine : on teste la generalisation du signeur 1.
