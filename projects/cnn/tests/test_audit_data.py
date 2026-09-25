import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from audit_data import audit, assign_groups, scan, proxy_key


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'images'
        self.counter = 0
        for split, groups in [('train', range(1, 9)), ('test', range(101, 104))]:
            for label in ['NORMAL', 'PNEUMONIA']:
                for group in groups:
                    for number in [1, 2]:
                        name = f'IM-{group:04d}-{number:04d}.png' if label=='NORMAL' else f'person{group}_bacteria_{number}.png'
                        self.create(split, label, name)

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, split, label, name):
        self.counter += 1
        p = self.root/split/label/name
        p.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('L',(16,16),self.counter)
        image.putpixel((3,7),255-self.counter)
        image.save(p)
        return p

    def rows(self):
        return assign_groups(scan(self.root)[0])

    def test_deterministic_split_and_source_preservation(self):
        a=audit(self.root,self.base/'a');b=audit(self.root,self.base/'b')
        self.assertEqual(a,b)
        self.assertEqual((self.base/'a/manifest.csv').read_bytes(),(self.base/'b/manifest.csv').read_bytes())
        self.assertEqual(a['original_counts'],{'train/NORMAL':16,'train/PNEUMONIA':16,'test/NORMAL':6,'test/PNEUMONIA':6})
        self.assertEqual(a['final_counts']['val/NORMAL'],4)
        self.assertEqual(a['final_counts']['val/PNEUMONIA'],4)
        self.assertFalse(a['patient_level_verified'])
        self.assertFalse(a['matches_notebook_total_counts'])

    def test_cross_split_proxy_quarantines_entire_group(self):
        self.create('test','PNEUMONIA','person1_virus_999.png')
        members=[r for r in self.rows() if r['proxy_group']=='pneumonia-person:1']
        self.assertEqual(len(members),3)
        self.assertTrue(all('cross_original_split_group' in r['reason'] for r in members))

    def test_pixel_duplicate_with_different_bytes_and_transitive_group(self):
        original=self.root/'train/NORMAL/IM-0001-0001.png'
        duplicate=self.root/'test/NORMAL/IM-0101-0001.png'
        with Image.open(original) as image:
            image.save(duplicate,compress_level=0)
        self.assertNotEqual(original.read_bytes(),duplicate.read_bytes())
        members=[r for r in self.rows() if r['proxy_group'] in ('im:1','im:101')]
        self.assertEqual(len(members),4)
        self.assertEqual(len({r['group_id'] for r in members}),1)
        self.assertTrue(all(r['status']=='quarantine' for r in members))

    def test_conflicting_labels_quarantine_both_groups(self):
        shutil.copyfile(self.root/'train/NORMAL/IM-0001-0001.png',self.root/'train/PNEUMONIA/person1_bacteria_1.png')
        members=[r for r in self.rows() if r['proxy_group'] in ('im:1','pneumonia-person:1')]
        self.assertTrue(all('conflicting_labels_group' in r['reason'] for r in members))

    def test_within_split_duplicate_retains_one(self):
        shutil.copyfile(self.root/'train/NORMAL/IM-0001-0001.png',self.root/'train/NORMAL/IM-0001-0002.png')
        members=[r for r in self.rows() if r['proxy_group']=='im:1']
        self.assertEqual(sum(r['status']=='eligible' for r in members),1)

    def test_corrupt_image_isolates_associated_group(self):
        (self.root/'train/NORMAL/IM-0001-0001.png').write_bytes(b'broken image')
        members=[r for r in self.rows() if r['proxy_group']=='im:1']
        self.assertTrue(all(r['status']=='quarantine' for r in members))

    def test_filename_suffix_and_namespace(self):
        self.assertEqual(proxy_key('person447_virus_921_1.jpeg', 'x'), ('pneumonia-person:447', True))
        self.assertNotEqual(proxy_key('IM-0001-0001.jpeg','x')[0], proxy_key('NORMAL2-IM-0001-0001.jpeg','y')[0])
        self.assertFalse(proxy_key('unknown.png','z')[1])

    def test_missing_directory_and_existing_report_fail(self):
        with self.assertRaises(ValueError):
            audit(self.base/'missing',self.base/'out')
        audit(self.root,self.base/'out')
        with self.assertRaises(ValueError):
            audit(self.root,self.base/'out')

    def test_metadata_ignored_and_unexpected_class_rejected(self):
        (self.root/'train/NORMAL/.DS_Store').write_bytes(b'metadata')
        a=audit(self.root,self.base/'out')
        self.assertIn('train/NORMAL/.DS_Store',a['ignored_files'])
        (self.root/'train/OTHER').mkdir()
        with self.assertRaises(ValueError):
            scan(self.root)


if __name__=='__main__':
    unittest.main()
