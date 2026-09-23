import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { FACE_SWAP_RESULTS } from "@/lib/face-swap-results";

export function ResearchMetrics() {
  return (
    <section className="results-section" aria-labelledby="results-heading">
      <p className="eyebrow">MEASURED RESULTS</p>
      <h2 id="results-heading">FACADE face-swap tests</h2>
      <p className="results-intro">
        We ran real face swaps on both the original and the FACADE-protected phone images. Both tables use the same FaceNet scoring
        rule: clean similarity at least 0.4, protected similarity below 0.4, and a drop of at least 0.1.
      </p>
      <p className="results-intro">
        48 fictional development images × 4 fixed recipients = 192 matched pairs per upload condition. Every clean control qualified,
        with no missing scores. The 95% intervals resample the 48 source images, keeping each image’s four recipients together.
      </p>

      {FACE_SWAP_RESULTS.map((engine) => (
        <div className="results-panel" key={engine.name}>
          <h3>FACADE · {engine.name}</h3>
          <Table>
            <TableCaption>{engine.caption}</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Upload condition</TableHead>
                <TableHead>Passed pairs</TableHead>
                <TableHead>Observed rate</TableHead>
                <TableHead>95% interval</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {engine.rows.map((row) => (
                <TableRow key={row.key}>
                  <TableCell>{row.label}</TableCell>
                  <TableCell>
                    {row.passes} / {row.validPairs}
                  </TableCell>
                  <TableCell>{row.rate}</TableCell>
                  <TableCell>{row.interval}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ))}

      <p className="results-limit">
        These are exploratory results from reused development images, not a real-world protection guarantee. SimSwap’s identity
        network helped optimize FACADE. Both swappers use the same scoring method here, but their native crop sizes and alignment
        differ. The tests use saved phone outputs; they are not a fresh population test of a later app build.
      </p>

      <div className="results-panel">
        <h3>Why earlier InSwapper numbers were different</h3>
        <p>
          Earlier MagFace and SFace scoring recorded 192 / 192 threshold crossings in each condition on these same InSwapper outputs.
          Those were different scoring rules. The FaceNet results above are lower, especially after compression. The disagreement
          shows that the measured result depends on how identity similarity is assessed; the earlier counts do not establish universal
          protection.
        </p>
      </div>

      <div className="results-next">
        <h3>PhotoGuard on this website</h3>
        <p>
          The 512 × 512, 200-step PhotoGuard encoder runs on the team’s private GPU server. A finished run shows that the service
          works; it does not show that a particular edit is blocked. The full PhotoGuard demo lets you compare an AI edit of the
          original with the same edit of the protected image. We have not yet measured how often it stops clothing edits.
        </p>
      </div>
    </section>
  );
}
